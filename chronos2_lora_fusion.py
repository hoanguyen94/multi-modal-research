"""End-to-end Chronos-2 low-rank adaptation and raw-text fusion components.

The pretrained Chronos-2 backbone is not replaced by a learned market
architecture. Its base weights remain frozen while LoRA or TopLoRA attention
adapters, the existing raw-text adapters/attention, and the directional fusion
head are optimized jointly with binary cross-entropy.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from latent_fusion import (
    RawEmbeddingStore,
    ResidualMLPBlock,
    UnifiedRawTextAttention,
    _early_stopping_reached,
    _fusion_batches,
    _raw_text_batch,
)
from model_config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MIN_EPOCHS,
    EARLY_STOPPING_PATIENCE,
)


class TopLoRALinear(nn.Module):
    """Token-wise projected low-rank update ``B diag(s_X) A``.

    The token gate follows the TopLoRA paper:
    ``s_X = exp(RMSNorm(Theta X))``. The pretrained linear map remains frozen,
    LoRA-B starts at zero, and LoRA-A and Theta use Kaiming initialization.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid TopLoRA rank, alpha, or dropout")
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        factory = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(
            base_layer.in_features, rank, bias=False, **factory
        )
        self.lora_B = nn.Linear(
            rank, base_layer.out_features, bias=False, **factory
        )
        self.toplora_projector = nn.Parameter(
            torch.empty(
                base_layer.in_features,
                rank,
                **factory,
            )
        )
        self.toplora_norm = nn.RMSNorm(rank, **factory)
        nn.init.kaiming_uniform_(
            self.lora_A.weight, a=math.sqrt(5)
        )
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(
            self.toplora_projector,
            a=math.sqrt(5),
            mode="fan_out",
        )

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(values)
        adapter_values = values.to(self.lora_A.weight.dtype)
        token_scale = torch.exp(
            self.toplora_norm(adapter_values @ self.toplora_projector)
        )
        update = self.lora_B(
            self.lora_A(self.lora_dropout(adapter_values)) * token_scale
        )
        return base + update.to(base.dtype) * self.scaling


def inject_toplora(
    model: nn.Module,
    *,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Replace matching frozen linear projections with TopLoRA layers."""
    targets = tuple(target_modules)
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and any(name.endswith(target) for target in targets)
        ):
            replacements.append((name, module))
    if not replacements:
        raise ValueError(
            f"No linear modules matched TopLoRA targets: {targets}"
        )
    replaced = []
    for name, module in replacements:
        parent = model
        components = name.split(".")
        for component in components[:-1]:
            parent = getattr(parent, component)
        setattr(
            parent,
            components[-1],
            TopLoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced.append(name)
    return replaced


@dataclass(frozen=True)
class ChronosContextStore:
    """Compact histories that materialize fixed-width contexts per mini-batch."""

    histories: dict[str, np.ndarray]
    locations: dict[int, tuple[str, int]]
    input_columns: tuple[str, ...]
    horizon: int
    lookback: int
    min_context: int

    @classmethod
    def from_features(
        cls,
        features: pl.DataFrame,
        input_columns: Sequence[str],
        *,
        horizon: int,
        lookback: int,
        min_context: int,
    ) -> "ChronosContextStore":
        columns = tuple(input_columns)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Chronos-2 input columns must be non-empty and unique")
        missing = sorted(set(columns) - set(features.columns))
        if missing:
            raise ValueError(f"Missing Chronos-2 context columns: {missing}")
        histories: dict[str, np.ndarray] = {}
        locations: dict[int, tuple[str, int]] = {}
        selected = ["row_id", "ticker", "date", *columns]
        for key, frame in features.select(selected).sort(
            ["ticker", "date"]
        ).partition_by("ticker", as_dict=True).items():
            ticker = key[0] if isinstance(key, tuple) else key
            histories[ticker] = (
                frame.select(columns).to_numpy().astype(np.float32).T
            )
            for position, row_id in enumerate(frame["row_id"].to_list()):
                integer_id = int(row_id)
                if integer_id in locations:
                    raise ValueError(f"Duplicate row_id in context data: {integer_id}")
                locations[integer_id] = (ticker, position)
        return cls(
            histories=histories,
            locations=locations,
            input_columns=columns,
            horizon=int(horizon),
            lookback=int(lookback),
            min_context=int(min_context),
        )

    def _slice(self, row_id: int) -> np.ndarray | None:
        location = self.locations.get(int(row_id))
        if location is None:
            raise ValueError(f"No Chronos-2 history for row_id={row_id}")
        ticker, position = location
        start = max(self.horizon, position + 1 - self.lookback)
        context = self.histories[ticker][:, start:position + 1]
        if (
            context.shape[1] < self.min_context
            or not np.isfinite(context).all()
        ):
            return None
        return context

    def valid_index(self, index: pl.DataFrame) -> pl.DataFrame:
        """Retain index rows for which a finite minimum-length context exists."""
        valid = [
            self._slice(int(row_id)) is not None
            for row_id in index["row_id"].to_list()
        ]
        return index.filter(pl.Series("_valid_context", valid))

    def batch(self, row_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Build right-aligned ``(B, n_variates, lookback)`` contexts."""
        result = np.full(
            (len(row_ids), len(self.input_columns), self.lookback),
            np.nan,
            dtype=np.float32,
        )
        for output_row, row_id in enumerate(row_ids):
            context = self._slice(int(row_id))
            if context is None:
                raise ValueError(f"Invalid Chronos-2 context for row_id={row_id}")
            result[output_row, :, -context.shape[1]:] = context
        return result


def assemble_raw_text_indices(
    index: pl.DataFrame,
    links: pl.DataFrame,
    stores: dict[str, RawEmbeddingStore],
    text_fields: Sequence[str],
) -> dict[str, np.ndarray]:
    """Align disk-backed raw embeddings to a model index without price caches."""
    fields = tuple(text_fields)
    if not fields:
        raise ValueError("At least one text field is required")
    if index.select(pl.col("row_id").is_duplicated().any()).item():
        raise ValueError("Model index contains duplicate row IDs")
    row_lookup = (
        index.select(pl.col("row_id").cast(pl.UInt64))
        .with_row_index("_row_index")
    )
    field_lookup = pl.DataFrame({
        "text_field": list(fields),
        "_field_index": np.arange(len(fields), dtype=np.int32),
    })
    scoped_links = (
        links.select([
            pl.col("row_id").cast(pl.UInt64), "text_field", "text_id",
        ])
        .unique()
        .join(row_lookup, on="row_id", how="inner", validate="m:1")
        .join(field_lookup, on="text_field", how="inner", validate="m:1")
    )
    result: dict[str, np.ndarray] = {}
    for family, store in stores.items():
        indices = np.full((index.height, len(fields)), -1, dtype=np.int64)
        aligned = scoped_links.join(
            store.index, on="text_id", how="inner", validate="m:1"
        )
        if aligned.height:
            duplicate = aligned.select(
                pl.struct(["_row_index", "_field_index"]).is_duplicated().any()
            ).item()
            if duplicate:
                raise ValueError(
                    f"Multiple {family} articles occupy one row/text-field slot"
                )
            indices[
                aligned["_row_index"].to_numpy(),
                aligned["_field_index"].to_numpy(),
            ] = aligned["embedding_row"].to_numpy()
        result[family] = np.ascontiguousarray(indices)
    return result


class Chronos2LoRATextFusion(nn.Module):
    """Low-rank-adapted Chronos-2 with the raw-text fusion head."""

    def __init__(
        self,
        chronos_model: nn.Module,
        *,
        n_variates: int,
        chronos_dim: int,
        covariate_dim: int,
        family_dims: dict[str, int],
        field_count: int,
        hidden_dim: int,
        text_dim: int,
        fusion_depth: int = 2,
        expansion: int = 2,
        text_attention_heads: int = 4,
        text_attention_layers: int = 1,
        dropout: float = 0.1,
        adapter_type: str = "lora",
    ):
        super().__init__()
        if n_variates < 1 or chronos_dim < 1 or covariate_dim < 1:
            raise ValueError("Chronos, variate, and covariate dimensions must be positive")
        self.chronos_model = chronos_model
        self.adapter_type = str(adapter_type)
        self.n_variates = int(n_variates)
        self.chronos_dim = int(chronos_dim)
        self.market_projection = nn.Sequential(
            nn.LayerNorm(n_variates * chronos_dim),
            nn.Linear(n_variates * chronos_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.covariate_projection = nn.Sequential(
            nn.LayerNorm(covariate_dim),
            nn.Linear(covariate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.market_norm = nn.LayerNorm(hidden_dim)
        self.text_attention = UnifiedRawTextAttention(
            family_dims,
            field_count,
            text_dim,
            hidden_dim,
            text_attention_heads,
            text_attention_layers,
            dropout,
        )
        self.fuse_input = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, expansion, dropout)
            for _ in range(fusion_depth)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def _core_model(self) -> nn.Module:
        getter = getattr(self.chronos_model, "get_base_model", None)
        return getter() if callable(getter) else self.chronos_model

    def encode_market(self, contexts: torch.Tensor) -> torch.Tensor:
        """Return differentiable, per-variate pooled Chronos encoder states."""
        if (
            contexts.ndim != 3
            or contexts.shape[1] != self.n_variates
        ):
            raise ValueError(
                "contexts must have shape "
                f"(batch, {self.n_variates}, history)"
            )
        batch_size, n_variates, history = contexts.shape
        flat = contexts.reshape(batch_size * n_variates, history)
        group_ids = torch.arange(
            batch_size, device=contexts.device, dtype=torch.long
        ).repeat_interleave(n_variates)
        core = self._core_model()
        (
            encoder_outputs,
            _,
            _,
            num_context_patches,
        ) = core.encode(
            context=flat,
            group_ids=group_ids,
        )
        hidden = encoder_outputs[0]
        if (
            hidden.ndim != 3
            or num_context_patches < 1
            or hidden.shape[1] < num_context_patches
        ):
            raise RuntimeError(
                f"Unexpected Chronos-2 encoder shape: {tuple(hidden.shape)}"
            )
        # Reproduce Chronos-2's context-patch validity mask. Fixed-width
        # mini-batches are left padded with NaN, and masked patch states must
        # not contribute to the pooled representation.
        scalar_mask = torch.isfinite(flat).to(flat.dtype)
        patched_mask = torch.nan_to_num(core.patch(scalar_mask), nan=0.0)
        valid_patches = patched_mask.sum(dim=-1) > 0
        context_hidden = hidden[:, :num_context_patches, :]
        if context_hidden.shape[:2] != valid_patches.shape:
            raise RuntimeError(
                "Chronos-2 context tokens and patch mask are misaligned"
            )
        pooled = (
            context_hidden
            * valid_patches.unsqueeze(-1).to(context_hidden.dtype)
        ).sum(dim=1)
        pooled = pooled / valid_patches.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(pooled.dtype)
        if pooled.shape[-1] != self.chronos_dim:
            raise RuntimeError("Chronos-2 hidden width differs from configuration")
        return pooled.reshape(batch_size, n_variates * self.chronos_dim)

    def forward(
        self,
        contexts: torch.Tensor,
        covariates: torch.Tensor,
        raw_articles: dict[str, torch.Tensor],
        article_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        market = self.market_projection(self.encode_market(contexts))
        market = self.market_norm(
            market + self.covariate_projection(covariates)
        )
        text, _ = self.text_attention(
            raw_articles, article_masks, market
        )
        fused = self.fuse_input(torch.cat([market, text], dim=1))
        for block in self.fusion_blocks:
            fused = block(fused)
        return self.classifier(self.output_norm(fused)).squeeze(1)


def load_chronos2_lora_fusion(
    *,
    model_id: str,
    device: str,
    n_variates: int,
    covariate_dim: int,
    family_dims: dict[str, int],
    field_count: int,
    hidden_dim: int,
    text_dim: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    fusion_depth: int,
    expansion: int,
    text_attention_heads: int,
    text_attention_layers: int,
    dropout: float,
    adapter_type: str = "lora",
) -> Chronos2LoRATextFusion:
    """Load Chronos-2 and inject LoRA or token-wise TopLoRA."""
    try:
        from chronos import Chronos2Pipeline
    except ImportError as error:
        raise ImportError(
            "Chronos-2 adapter fusion requires chronos-forecasting>=2.1.0."
        ) from error
    if adapter_type not in {"lora", "toplora"}:
        raise ValueError("adapter_type must be 'lora' or 'toplora'")
    if lora_rank < 1 or lora_alpha < 1 or not 0.0 <= lora_dropout < 1.0:
        raise ValueError("Invalid LoRA rank, alpha, or dropout")
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
    core = pipeline.model
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    targets = (
        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
    )
    if adapter_type == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            raise ImportError(
                "Standard LoRA requires peft>=0.18.1."
            ) from error
        adapted_model = get_peft_model(
            core,
            LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(targets),
            ),
        )
    else:
        inject_toplora(
            core,
            target_modules=targets,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )
        adapted_model = core
    trainable_chronos = [
        name for name, parameter in adapted_model.named_parameters()
        if parameter.requires_grad
    ]
    allowed_markers = (
        ("lora_",)
        if adapter_type == "lora"
        else ("lora_", "toplora_")
    )
    if not trainable_chronos or any(
        not any(marker in name for marker in allowed_markers)
        for name in trainable_chronos
    ):
        raise RuntimeError(
            "Chronos-2 contains unexpected trainable base parameters"
        )
    chronos_dim = int(getattr(core, "model_dim"))
    model = Chronos2LoRATextFusion(
        adapted_model,
        n_variates=n_variates,
        chronos_dim=chronos_dim,
        covariate_dim=covariate_dim,
        family_dims=family_dims,
        field_count=field_count,
        hidden_dim=hidden_dim,
        text_dim=text_dim,
        fusion_depth=fusion_depth,
        expansion=expansion,
        text_attention_heads=text_attention_heads,
        text_attention_layers=text_attention_layers,
        dropout=dropout,
        adapter_type=adapter_type,
    )
    return model.to(device)


def _copy_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters).intersection(state) != set(state):
        raise ValueError("Saved trainable state does not match model parameters")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))


def _optimizer(
    model: Chronos2LoRATextFusion,
    *,
    learning_rate: float,
    lora_learning_rate: float,
    adapter_learning_rate_multiplier: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    chronos_adapter_parameters = []
    text_adapter_parameters, head_parameters = [], []
    adapter_ids = {
        id(parameter)
        for parameter in model.text_attention.adapters.parameters()
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "chronos_model" in name and "lora_" in name:
            chronos_adapter_parameters.append(parameter)
        elif id(parameter) in adapter_ids:
            text_adapter_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    if not chronos_adapter_parameters:
        raise RuntimeError(
            "No trainable Chronos-2 adapter parameters were found"
        )
    return torch.optim.AdamW([
        {"params": head_parameters, "lr": learning_rate},
        {
            "params": text_adapter_parameters,
            "lr": learning_rate * adapter_learning_rate_multiplier,
        },
        {"params": chronos_adapter_parameters, "lr": lora_learning_rate},
    ], weight_decay=weight_decay)


@torch.no_grad()
def predict_lora_fusion(
    model: Chronos2LoRATextFusion,
    *,
    index: pl.DataFrame,
    contexts: ChronosContextStore,
    covariates: np.ndarray,
    text_indices: dict[str, np.ndarray],
    stores: dict[str, RawEmbeddingStore],
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = np.empty(index.height, dtype=np.float32)
    row_ids = index["row_id"].to_numpy()
    for batch in _fusion_batches(index.height, batch_size):
        articles, masks = _raw_text_batch(
            text_indices, stores, batch, device
        )
        context = torch.from_numpy(contexts.batch(row_ids[batch])).to(device)
        covariate = torch.from_numpy(covariates[batch]).to(device)
        scores[batch] = torch.sigmoid(
            model(context, covariate, articles, masks)
        ).float().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError(
            "Chronos-2 adapter fusion produced non-finite scores"
        )
    return scores


def fit_lora_fusion(
    model: Chronos2LoRATextFusion,
    *,
    train_index: pl.DataFrame,
    contexts: ChronosContextStore,
    train_covariates: np.ndarray,
    train_text_indices: dict[str, np.ndarray],
    stores: dict[str, RawEmbeddingStore],
    train_target: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_learning_rate: float,
    adapter_learning_rate_multiplier: float,
    weight_decay: float,
    seed: int,
    validation_index: pl.DataFrame | None = None,
    validation_covariates: np.ndarray | None = None,
    validation_text_indices: dict[str, np.ndarray] | None = None,
    validation_target: np.ndarray | None = None,
    early_stopping_min_epochs: int = EARLY_STOPPING_MIN_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
) -> tuple[Chronos2LoRATextFusion, list[dict[str, float]]]:
    """Jointly train the Chronos adapter and multimodal decision layers."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if early_stopping_min_epochs < 1:
        raise ValueError("early_stopping_min_epochs must be positive")
    if epochs < early_stopping_min_epochs:
        raise ValueError(
            "epochs must be at least early_stopping_min_epochs"
        )
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta cannot be negative")
    if len(train_index) != len(train_target) or len(train_covariates) != len(train_target):
        raise ValueError("Training arrays are not aligned")
    has_validation = validation_index is not None
    validation_values = (
        validation_covariates,
        validation_text_indices,
        validation_target,
    )
    if has_validation != all(value is not None for value in validation_values):
        raise ValueError("Validation inputs must be supplied together")
    optimizer = _optimizer(
        model,
        learning_rate=learning_rate,
        lora_learning_rate=lora_learning_rate,
        adapter_learning_rate_multiplier=adapter_learning_rate_multiplier,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    row_ids = train_index["row_id"].to_numpy()
    history: list[dict[str, float]] = []
    best_state = None
    best_epoch = None
    best_validation = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for batch in _fusion_batches(
            len(train_target), batch_size, rng=rng
        ):
            articles, masks = _raw_text_batch(
                train_text_indices, stores, batch, device
            )
            context = torch.from_numpy(
                contexts.batch(row_ids[batch])
            ).to(device)
            covariate = torch.from_numpy(train_covariates[batch]).to(device)
            target = torch.from_numpy(
                train_target[batch].astype(np.float32, copy=False)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context, covariate, articles, masks)
            per_example = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = per_example.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Chronos-2 adapter loss became non-finite"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            loss_sum += float(per_example.detach().sum().cpu())
            example_count += len(batch)
        row = {
            "epoch": float(epoch),
            "train_bce": loss_sum / max(example_count, 1),
        }
        if has_validation:
            score = predict_lora_fusion(
                model,
                index=validation_index,
                contexts=contexts,
                covariates=validation_covariates,
                text_indices=validation_text_indices,
                stores=stores,
                device=device,
                batch_size=batch_size,
            )
            truth = validation_target.astype(np.float32, copy=False)
            per_example = F.binary_cross_entropy(
                torch.from_numpy(score),
                torch.from_numpy(truth),
                reduction="mean",
            )
            row["validation_bce"] = float(per_example)
            if (
                epoch >= early_stopping_min_epochs
                and row["validation_bce"]
                < best_validation - early_stopping_min_delta
            ):
                best_validation = row["validation_bce"]
                best_epoch = epoch
                best_state = _copy_trainable_state(model)
                stale_epochs = 0
            elif epoch >= early_stopping_min_epochs:
                stale_epochs += 1
        history.append(row)
        if has_validation and _early_stopping_reached(
            epoch=epoch,
            epochs_without_improvement=stale_epochs,
            min_epochs=early_stopping_min_epochs,
            patience=early_stopping_patience,
        ):
            break
    if best_state is not None:
        _restore_trainable_state(model, best_state)
    if best_epoch is None:
        best_epoch = len(history)
    for row in history:
        row["best_epoch"] = float(best_epoch)
        row["is_best_epoch"] = float(row["epoch"] == best_epoch)
        row["stopped_early"] = float(len(history) < epochs)
    return model, history


def save_lora_fusion(
    model: Chronos2LoRATextFusion,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Persist the compact adapter separately from the multimodal head."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if model.adapter_type == "lora":
        model.chronos_model.save_pretrained(
            output_dir / "chronos2_lora_adapter"
        )
    elif model.adapter_type == "toplora":
        adapter_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if (
                name.startswith("chronos_model.")
                and parameter.requires_grad
            )
        }
        if not adapter_state:
            raise RuntimeError("No trainable TopLoRA state was found")
        torch.save(
            adapter_state,
            output_dir / "chronos2_toplora_adapter.pt",
        )
    else:
        raise ValueError(f"Unsupported adapter type: {model.adapter_type}")
    head_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("chronos_model.")
    }
    torch.save(head_state, output_dir / "multimodal_head.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str)
    )
    gc.collect()


def load_saved_lora_fusion(
    artifact_dir: Path,
    *,
    device: str,
) -> tuple[Chronos2LoRATextFusion, dict]:
    """Reconstruct a saved low-rank adapter and multimodal decision head."""
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    adapter_type = metadata.get("adapter_type", "lora")
    common = dict(
        model_id=metadata["model_id"],
        device="cpu",
        n_variates=int(metadata["n_variates"]),
        covariate_dim=int(metadata["covariate_dim"]),
        family_dims={
            str(key): int(value)
            for key, value in metadata["family_dims"].items()
        },
        field_count=int(metadata["field_count"]),
        hidden_dim=int(metadata["hidden_dim"]),
        text_dim=int(metadata["text_dim"]),
        fusion_depth=int(metadata["fusion_depth"]),
        expansion=int(metadata["expansion"]),
        text_attention_heads=int(metadata["text_attention_heads"]),
        text_attention_layers=int(metadata["text_attention_layers"]),
        dropout=float(metadata["fusion_dropout"]),
    )
    if adapter_type == "lora":
        try:
            from chronos import Chronos2Pipeline
            from peft import PeftModel
        except ImportError as error:
            raise ImportError(
                "Loading LoRA artifacts requires chronos-forecasting and peft."
            ) from error
        pipeline = Chronos2Pipeline.from_pretrained(
            metadata["model_id"], device_map="cpu"
        )
        lora = PeftModel.from_pretrained(
            pipeline.model,
            artifact_dir / "chronos2_lora_adapter",
            is_trainable=False,
        )
        model = Chronos2LoRATextFusion(
            lora,
            n_variates=common["n_variates"],
            chronos_dim=int(getattr(pipeline.model, "model_dim")),
            covariate_dim=common["covariate_dim"],
            family_dims=common["family_dims"],
            field_count=common["field_count"],
            hidden_dim=common["hidden_dim"],
            text_dim=common["text_dim"],
            fusion_depth=common["fusion_depth"],
            expansion=common["expansion"],
            text_attention_heads=common["text_attention_heads"],
            text_attention_layers=common["text_attention_layers"],
            dropout=common["dropout"],
            adapter_type="lora",
        )
    elif adapter_type == "toplora":
        model = load_chronos2_lora_fusion(
            **common,
            lora_rank=int(metadata["lora_rank"]),
            lora_alpha=int(metadata["lora_alpha"]),
            lora_dropout=float(metadata["lora_dropout"]),
            adapter_type="toplora",
        )
        adapter_state = torch.load(
            artifact_dir / "chronos2_toplora_adapter.pt",
            map_location="cpu",
            weights_only=True,
        )
        _restore_trainable_state(model, adapter_state)
        for parameter in model.chronos_model.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError(f"Unsupported saved adapter type: {adapter_type}")
    head_state = torch.load(
        artifact_dir / "multimodal_head.pt",
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    if unexpected or any(
        not name.startswith("chronos_model.") for name in missing
    ):
        raise RuntimeError(
            "Saved multimodal head is incompatible with its metadata"
        )
    return model.to(device).eval(), metadata
