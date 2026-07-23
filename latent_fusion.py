"""Memory-conscious helpers for frozen-price, covariate, and text fusion.

The notebook owns TimesFM loading and hidden-state extraction.  This module
trains one small text encoder per original embedding family, projects articles
in streaming parquet batches, pools articles within each stock-date, and fits a
residual market MLP followed by the paper draft's simple-concatenation fusion.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from utils import directional_classification_metrics, probability_to_price


def parquet_embedding_dim(path: Path) -> int:
    """Return the number of emb_* columns and validate the parquet footer."""
    schema = pq.ParquetFile(path).schema_arrow
    columns = [name for name in schema.names if name.startswith("emb_")]
    if "text_id" not in schema.names or not columns:
        raise ValueError(f"{path} must contain text_id and emb_* columns")
    return len(columns)


def _nonfinite_row_count(frame: pl.DataFrame, columns: Sequence[str]) -> int:
    """Count rows containing NaN or infinity without one large NumPy copy."""
    if not columns:
        return 0
    return int(frame.select(
        pl.any_horizontal([
            pl.col(column).is_nan() | pl.col(column).is_infinite()
            for column in columns
        ]).sum()
    ).item())


def _require_finite(name: str, values: np.ndarray, batch_rows: int = 4096) -> None:
    """Raise a targeted error for non-finite model inputs without a huge mask."""
    for start in range(0, len(values), batch_rows):
        chunk = values[start:start + batch_rows]
        if not np.isfinite(chunk).all():
            invalid = int((~np.isfinite(chunk)).sum())
            raise ValueError(
                f"{name} contains at least {invalid:,} NaN/inf values near rows "
                f"{start}:{start + len(chunk)}"
            )


class TextCoder(nn.Module):
    """h_T = LayerNorm(GELU(W_T z_T + b_T))."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.projection = nn.Linear(input_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(F.gelu(self.projection(values))))


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP block with a wider inner layer."""

    def __init__(self, hidden_dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # Begin close to the identity map, then learn residual corrections.
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.mlp(self.norm(values))


class CovariateResidualFusion(nn.Module):
    """Fuse a residual market MLP with family-specific text representations.

    The frozen TimesFM vector is concatenated with fold-standardized engineered
    covariates first.  A residual MLP turns that joint numeric input into the
    market representation.  Original text-embedding families are encoded and
    pooled separately, then meet the market representation only at the final
    simple-concatenation residual head.
    """

    def __init__(
        self,
        price_dim: int,
        covariate_dim: int,
        text_dim: int,
        family_count: int,
        hidden_dim: int,
        market_depth: int = 2,
        fusion_depth: int = 2,
        expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(market_depth, fusion_depth) < 0:
            raise ValueError("residual depths must be non-negative")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        self.price_dim = int(price_dim)
        self.covariate_dim = int(covariate_dim)
        self.text_dim = int(text_dim)
        self.family_count = int(family_count)
        self.hidden_dim = int(hidden_dim)
        self.market_depth = int(market_depth)
        self.fusion_depth = int(fusion_depth)
        self.expansion = int(expansion)
        # Family vectors are kept separate until this learned text pooling map.
        self.text_pool = nn.Sequential(
            nn.Linear(family_count * text_dim + family_count, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
        )
        self.market_input = nn.Sequential(
            nn.LayerNorm(price_dim + covariate_dim),
            nn.Linear(price_dim + covariate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.market_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, expansion, dropout)
            for _ in range(market_depth)
        ])
        self.market_norm = nn.LayerNorm(hidden_dim)
        self.fuse_input = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, expansion, dropout)
            for _ in range(fusion_depth)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        price: torch.Tensor,
        covariates: torch.Tensor,
        family_text: torch.Tensor,
        family_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_text = family_text * family_mask.unsqueeze(-1)
        text_input = torch.cat(
            [masked_text.flatten(start_dim=1), family_mask], dim=1
        )
        h_text = self.text_pool(text_input)
        market = self.market_input(torch.cat([price, covariates], dim=1))
        for block in self.market_blocks:
            market = block(market)
        market = self.market_norm(market)
        fused = self.fuse_input(torch.cat([market, h_text], dim=1))
        for block in self.residual_blocks:
            fused = block(fused)
        return self.classifier(self.output_norm(fused)).squeeze(1)


def make_text_supervision(links: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Make soft article labels while giving every stock-date total weight one."""
    labeled_links = (
        links.select(["row_id", "text_id"])
        .unique()
        .join(targets.select(["row_id", "target_up"]), on="row_id", how="inner")
        .filter(pl.col("target_up").is_not_null())
    )
    row_counts = labeled_links.group_by("row_id").len().rename({"len": "article_count"})
    return (
        labeled_links.join(row_counts, on="row_id", how="left")
        .with_columns((1.0 / pl.col("article_count")).alias("row_weight"))
        .group_by("text_id")
        .agg(
            (pl.col("target_up") * pl.col("row_weight")).sum().alias("weighted_target"),
            pl.col("row_weight").sum().alias("sample_weight"),
        )
        .with_columns(
            (pl.col("weighted_target") / pl.col("sample_weight")).cast(pl.Float32).alias("soft_target"),
            pl.col("sample_weight").cast(pl.Float32),
        )
        .select(["text_id", "soft_target", "sample_weight"])
    )


def train_text_coder_streaming(
    embedding_path: Path,
    supervision: pl.DataFrame,
    latent_dim: int,
    device: str,
    epochs: int = 3,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> tuple[TextCoder, list[dict[str, float]]]:
    """Train directly from original embedding columns without materializing them."""
    input_dim = parquet_embedding_dim(embedding_path)
    encoder = TextCoder(input_dim, latent_dim).to(device)
    auxiliary_head = nn.Linear(latent_dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(auxiliary_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lookup = {
        text_id: (float(target), float(weight))
        for text_id, target, weight in supervision.iter_rows()
    }
    columns = ["text_id"] + [f"emb_{j}" for j in range(input_dim)]
    history: list[dict[str, float]] = []
    parquet_file = pq.ParquetFile(embedding_path)
    for epoch in range(1, epochs + 1):
        encoder.train(); auxiliary_head.train()
        weighted_loss = 0.0; total_weight = 0.0; used = 0
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            ids = record_batch.column(0).to_pylist()
            selected = [(j, lookup[text_id]) for j, text_id in enumerate(ids) if text_id in lookup]
            if not selected:
                continue
            positions = np.fromiter((item[0] for item in selected), dtype=np.int64)
            values = np.column_stack([
                record_batch.column(j).to_numpy(zero_copy_only=False)[positions]
                for j in range(1, input_dim + 1)
            ]).astype(np.float32, copy=False)
            target = np.fromiter((item[1][0] for item in selected), dtype=np.float32)
            weight = np.fromiter((item[1][1] for item in selected), dtype=np.float32)
            x = torch.from_numpy(values).to(device)
            y = torch.from_numpy(target).to(device)
            w = torch.from_numpy(weight).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = auxiliary_head(encoder(x)).squeeze(1)
            loss_each = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (loss_each * w).sum() / w.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            weighted_loss += float((loss_each.detach() * w).sum().cpu())
            total_weight += float(w.sum().cpu())
            used += len(selected)
        history.append({
            "epoch": float(epoch),
            "weighted_bce": weighted_loss / max(total_weight, 1e-8),
            "articles_used": float(used),
        })
        if used == 0:
            raise ValueError(
                f"No supervised text_id from {embedding_path} matched the training links"
            )
    return encoder.cpu().eval(), history


@torch.inference_mode()
def project_articles(
    encoder: TextCoder,
    embedding_path: Path,
    needed_text_ids: set[str],
    output_path: Path,
    device: str,
    batch_size: int = 1024,
) -> None:
    """Stream original vectors through a trained coder and save compact tokens."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_dim, latent_dim = encoder.input_dim, encoder.latent_dim
    columns = ["text_id"] + [f"emb_{j}" for j in range(input_dim)]
    names = ["text_id"] + [f"latent_{j:04d}" for j in range(latent_dim)]
    schema = pa.schema([pa.field("text_id", pa.string())] + [
        pa.field(name, pa.float32()) for name in names[1:]
    ])
    encoder = encoder.to(device).eval()
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    try:
        for record_batch in pq.ParquetFile(embedding_path).iter_batches(
            batch_size=batch_size, columns=columns
        ):
            ids = record_batch.column(0).to_pylist()
            positions = np.fromiter(
                (j for j, text_id in enumerate(ids) if text_id in needed_text_ids),
                dtype=np.int64,
            )
            if not len(positions):
                continue
            values = np.column_stack([
                record_batch.column(j).to_numpy(zero_copy_only=False)[positions]
                for j in range(1, input_dim + 1)
            ]).astype(np.float32, copy=False)
            latent = encoder(torch.from_numpy(values).to(device)).cpu().numpy().astype(np.float32)
            arrays = [pa.array([ids[j] for j in positions], type=pa.string())]
            arrays.extend(pa.array(latent[:, j]) for j in range(latent_dim))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    finally:
        writer.close()
        encoder.cpu()


def pool_projected_articles(
    links: pl.DataFrame,
    projected_path: Path,
    family: str,
    output_path: Path,
) -> pl.DataFrame:
    """Mean-pool articles within a row and within one embedding family only."""
    latent_columns = [
        name for name in pq.ParquetFile(projected_path).schema_arrow.names
        if name.startswith("latent_")
    ]
    pooled = (
        links.lazy().select(["row_id", "text_id"]).unique()
        .join(pl.scan_parquet(projected_path), on="text_id", how="inner")
        .group_by("row_id")
        .agg([pl.col(name).mean().cast(pl.Float32) for name in latent_columns])
        .rename({name: f"{family}_{name}" for name in latent_columns})
        .collect(engine="streaming")
        .sort("row_id")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.write_parquet(output_path, compression="zstd")
    return pooled


def fit_fusion_model(
    price: np.ndarray,
    covariates: np.ndarray,
    family_text: np.ndarray,
    family_mask: np.ndarray,
    target: np.ndarray,
    device: str,
    hidden_dim: int = 256,
    market_depth: int = 2,
    fusion_depth: int = 2,
    expansion: int = 2,
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> tuple[CovariateResidualFusion, list[dict[str, float]]]:
    """Fit the fusion map and classifier; pretrained/text encoders stay frozen."""
    _require_finite("fusion-training price latents", price)
    _require_finite("fusion-training covariates", covariates)
    _require_finite("fusion-training text latents", family_text)
    _require_finite("fusion-training family mask", family_mask)
    _require_finite("fusion-training targets", target)
    torch.manual_seed(seed)
    model = CovariateResidualFusion(
        price.shape[1], covariates.shape[1], family_text.shape[2],
        family_text.shape[1], hidden_dim, market_depth=market_depth,
        fusion_depth=fusion_depth, expansion=expansion, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        order = rng.permutation(len(target))
        for start in range(0, len(target), batch_size):
            indices = order[start:start + batch_size]
            p = torch.from_numpy(price[indices]).to(device)
            c = torch.from_numpy(covariates[indices]).to(device)
            t = torch.from_numpy(family_text[indices]).to(device)
            m = torch.from_numpy(family_mask[indices]).to(device)
            y = torch.from_numpy(target[indices].astype(np.float32, copy=False)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(p, c, t, m), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Fusion loss became non-finite in epoch {epoch}; "
                    "inspect input scales and lower the learning rate"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch), "bce": float(np.mean(losses))})
    return model.cpu().eval(), history


@torch.inference_mode()
def predict_fusion(
    model: CovariateResidualFusion,
    price: np.ndarray,
    covariates: np.ndarray,
    family_text: np.ndarray,
    family_mask: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    _require_finite("fusion-inference price latents", price)
    _require_finite("fusion-inference covariates", covariates)
    _require_finite("fusion-inference text latents", family_text)
    _require_finite("fusion-inference family mask", family_mask)
    model = model.to(device).eval()
    pieces = []
    for start in range(0, len(price), batch_size):
        stop = start + batch_size
        logits = model(
            torch.from_numpy(price[start:stop]).to(device),
            torch.from_numpy(covariates[start:stop]).to(device),
            torch.from_numpy(family_text[start:stop]).to(device),
            torch.from_numpy(family_mask[start:stop]).to(device),
        )
        pieces.append(torch.sigmoid(logits).cpu().numpy())
    model.cpu()
    result = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
    _require_finite("fusion probabilities", result)
    return result


def save_torch_model(path: Path, model: nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def _prepared_history_lookup(
    features: pl.DataFrame,
    target_column: str,
) -> tuple[dict, dict]:
    histories, positions = {}, {}
    for ticker, frame in features.sort(["ticker", "date"]).partition_by(
        "ticker", as_dict=True
    ).items():
        key = ticker[0] if isinstance(ticker, tuple) else ticker
        dates = frame["date"].to_list()
        histories[key] = {
            "target": frame[target_column].to_numpy().astype(np.float64),
        }
        positions[key] = {date: index for index, date in enumerate(dates)}
    return histories, positions


def _context_record(
    history: dict,
    position: int,
    horizon: int,
    lookback: int,
    min_context: int,
) -> dict | None:
    start = max(horizon, position + 1 - lookback)
    target = history["target"][start:position + 1].astype(np.float32)
    if len(target) < min_context or np.any(~np.isfinite(target)):
        return None
    return {"target": target}


def _load_frozen_timesfm(model_id: str, device: str):
    import timesfm

    wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_id, torch_compile=False
    )
    wrapper.model.to(device).eval()
    wrapper.model.device = torch.device(device)
    for parameter in wrapper.model.parameters():
        parameter.requires_grad_(False)
    return wrapper


@torch.inference_mode()
def _pooled_timesfm_hidden(
    wrapper,
    records: Sequence[dict],
    lookback: int,
    device: str,
) -> np.ndarray:
    from timesfm.torch.util import revin, update_running_stats

    module = wrapper.model
    patch_length = module.p
    series_by_row = [
        np.asarray(record["target"][-lookback:], dtype=np.float32)
        for record in records
    ]
    groups: dict[int, list[int]] = {}
    for row, series in enumerate(series_by_row):
        if not len(series):
            raise ValueError("Cannot extract a TimesFM latent from an empty context")
        patch_count = int(np.ceil(len(series) / patch_length))
        groups.setdefault(patch_count, []).append(row)

    # A shared LOOKBACK-width tensor creates fully padded prefix patches for
    # short histories. Those tokens can contaminate later valid transformer
    # states with NaNs. Grouping by actual patch count retains batching while
    # ensuring that every patch contains at least one observation.
    pooled_by_row: list[np.ndarray | None] = [None] * len(records)
    for patch_count, row_indices in groups.items():
        context_width = patch_count * patch_length
        values = np.zeros((len(row_indices), context_width), dtype=np.float32)
        masks = np.ones((len(row_indices), context_width), dtype=bool)
        for group_row, original_row in enumerate(row_indices):
            series = series_by_row[original_row]
            values[group_row, -len(series):] = series
            masks[group_row, -len(series):] = False

        inputs = torch.from_numpy(values).to(device)
        input_masks = torch.from_numpy(masks).to(device)
        valid = (~input_masks).to(inputs.dtype)
        count = valid.sum(1, keepdim=True).clamp_min(1.0)
        mean = (inputs * valid).sum(1, keepdim=True) / count
        variance = (((inputs - mean) ** 2) * valid).sum(1, keepdim=True) / count
        inputs = revin(inputs, mean, variance.sqrt(), reverse=False)
        inputs = torch.where(input_masks, 0.0, inputs)
        patched_inputs = inputs.reshape(len(row_indices), patch_count, patch_length)
        patched_masks = input_masks.reshape(len(row_indices), patch_count, patch_length)

        n = torch.zeros(len(row_indices), device=inputs.device)
        mu = torch.zeros_like(n)
        sigma = torch.zeros_like(n)
        patch_mu, patch_sigma = [], []
        for patch in range(patch_count):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, patch], patched_masks[:, patch]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        normalized = revin(
            patched_inputs,
            torch.stack(patch_mu, 1),
            torch.stack(patch_sigma, 1),
            reverse=False,
        )
        normalized = torch.where(patched_masks, 0.0, normalized)
        (_, hidden_tokens, _, _), _ = module(normalized, patched_masks)  # B, M, D
        valid_tokens = (~patched_masks).any(-1)  # B, M
        masked_hidden = torch.where(
            valid_tokens.unsqueeze(-1), hidden_tokens, torch.zeros_like(hidden_tokens)
        )
        if not torch.isfinite(masked_hidden).all():
            raise ValueError(
                "TimesFM produced a non-finite hidden state after removing "
                "fully padded patches"
            )
        pooled = masked_hidden.sum(1)
        pooled = pooled / valid_tokens.sum(1, keepdim=True).clamp_min(1).to(pooled.dtype)
        pooled_numpy = pooled.float().cpu().numpy()
        for group_row, original_row in enumerate(row_indices):
            pooled_by_row[original_row] = pooled_numpy[group_row]

    if any(value is None for value in pooled_by_row):
        raise RuntimeError("TimesFM pooling did not return every requested row")
    return np.stack(pooled_by_row).astype(np.float32, copy=False)


def generate_timesfm_price_latents(
    *,
    split: str,
    prepared_features: pl.DataFrame,
    origins: pl.DataFrame,
    cache_path: Path,
    model_id: str,
    device: str,
    horizon: int,
    lookback: int,
    min_context: int,
    batch_size: int = 16,
    run_extraction: bool = True,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Cache mean-pooled TimesFM final hidden tokens from prepared feature rows."""
    if cache_path.exists() and not force_refresh:
        cached = pl.read_parquet(cache_path)
        latent_columns = [
            column for column in cached.columns if column.startswith("price_latent_")
        ]
        invalid_rows = _nonfinite_row_count(cached, latent_columns)
        if invalid_rows == 0:
            return cached
        if not run_extraction:
            raise ValueError(
                f"{cache_path} contains {invalid_rows:,} rows with non-finite "
                "TimesFM latents; enable extraction to rebuild it"
            )
        print(
            f"Rebuilding {cache_path}: found {invalid_rows:,} rows with "
            "non-finite TimesFM latents."
        )
    if not run_extraction:
        raise FileNotFoundError(
            f"{cache_path} is absent; enable TimesFM latent extraction"
        )
    histories, positions = _prepared_history_lookup(prepared_features, "ret_20")
    wrapper = _load_frozen_timesfm(model_id, device)
    parts = []
    for batch in origins.iter_slices(batch_size):
        kept, records = [], []
        for row in batch.iter_rows(named=True):
            record = _context_record(
                histories[row["ticker"]],
                positions[row["ticker"]][row["date"]],
                horizon,
                lookback,
                min_context,
            )
            if record is not None:
                kept.append(row)
                records.append(record)
        if not records:
            continue
        latent = _pooled_timesfm_hidden(wrapper, records, lookback, device)
        identity = pl.DataFrame({
            "row_id": [row["row_id"] for row in kept],
            "date": [row["date"] for row in kept],
            "ticker": [row["ticker"] for row in kept],
        })
        parts.append(pl.concat([
            identity,
            pl.DataFrame(
                latent,
                schema=[f"price_latent_{j:04d}" for j in range(latent.shape[1])],
            ),
        ], how="horizontal"))
    if not parts:
        raise RuntimeError(f"No valid {split} TimesFM contexts were created")
    result = pl.concat(parts).sort("row_id")
    latent_columns = [
        column for column in result.columns if column.startswith("price_latent_")
    ]
    invalid_rows = _nonfinite_row_count(result, latent_columns)
    if invalid_rows:
        raise ValueError(
            f"Refusing to cache {invalid_rows:,} rows with non-finite TimesFM latents"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(cache_path, compression="zstd")
    del wrapper
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result


def _load_text_coder(path: Path, input_dim: int, latent_dim: int) -> TextCoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    coder = TextCoder(input_dim, latent_dim)
    coder.load_state_dict(checkpoint["state_dict"])
    return coder.eval()


def _scope_text_latents(
    scope: str,
    scope_dir: Path,
    data_dir: Path,
    train_links: pl.DataFrame,
    inference_links: pl.DataFrame,
    fit_targets: pl.DataFrame,
    families: Sequence[str],
    latent_dim: int,
    device: str,
    epochs: int,
    batch_size: int,
    force_refresh: bool,
    run_training: bool,
) -> dict[str, Path]:
    """Fit scope-specific coders and return pooled latent paths for all rows."""
    scope_dir.mkdir(parents=True, exist_ok=True)
    fit_links = train_links.join(
        fit_targets.select("row_id"), on="row_id", how="semi"
    ) # retain only rows in train_links that have row_id in fit_targets
    combined_links = pl.concat(
        [fit_links, inference_links], how="vertical_relaxed"
    ).unique(["row_id", "text_id"])
    needed_text_ids = set(combined_links["text_id"].to_list())
    supervision = make_text_supervision(fit_links, fit_targets)
    row_paths: dict[str, Path] = {}
    for family in families:
        embedding_path = data_dir / f"{family}_textemb.parquet"
        input_dim = parquet_embedding_dim(embedding_path)
        coder_path = scope_dir / "text_coders" / f"{family}_text_coder.pt"
        history_path = scope_dir / "text_coders" / f"{family}_training_history.csv"
        if coder_path.exists() and not force_refresh:
            coder = _load_text_coder(coder_path, input_dim, latent_dim)
        else:
            if not run_training:
                raise FileNotFoundError(
                    f"{coder_path} is absent; enable text-coder training to create it"
                )
            coder, history = train_text_coder_streaming(
                embedding_path,
                supervision,
                latent_dim,
                device,
                epochs=epochs,
                batch_size=batch_size,
            )
            save_torch_model(
                coder_path,
                coder,
                {
                    "scope": scope,
                    "family": family,
                    "source": str(embedding_path),
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "fit_row_count": fit_targets.height,
                    "article_aggregation": "mean_after_coder",
                },
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(history).write_csv(history_path)
        projected_path = (
            scope_dir / "projected_text_tokens" /
            f"{family}_projected_original_embeddings.parquet"
        )
        if not projected_path.exists() or force_refresh:
            project_articles(
                coder,
                embedding_path,
                needed_text_ids,
                projected_path,
                device,
                batch_size,
            )
        row_path = scope_dir / "row_text_latents" / f"{family}_article_mean_latent.parquet"
        if not row_path.exists() or force_refresh:
            pool_projected_articles(combined_links, projected_path, family, row_path)
        row_paths[family] = row_path
        del coder
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    return row_paths


def _assemble_fusion_arrays(
    row_ids: pl.DataFrame,
    price_latents: pl.DataFrame,
    targets: pl.DataFrame,
    row_text_paths: dict[str, Path],
    families: Sequence[str],
    require_target: bool = True,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align price, text, and labels for one fusion-model partition.

    Parameters
    ----------
    row_ids:
        The rows belonging to the requested partition, such as one fold's
        training or validation IDs. Rows outside this table are excluded.
    price_latents:
        Frozen TimesFM representations. It must contain ``row_id``, ``date``,
        ``ticker``, and columns named ``price_latent_*``.
    targets:
        Direction labels keyed by ``row_id``. A null label is allowed for test
        rows when ``require_target=False``.
    row_text_paths:
        One parquet path per embedding family. Each table contains a pooled
        stock-date representation named ``<family>_latent_*``.
    families:
        Ordered families included in this model variant. Their order becomes
        the family axis of the returned text tensor.
    require_target:
        If true, discard rows without ``target_up``. This is enabled for fold
        training/validation and disabled when producing final test scores.

    Returns
    -------
    index:
        Polars table containing ``row_id``, ``date``, ``ticker``, and label;
        shape ``(N, 4)``.
    price:
        Contiguous float32 TimesFM matrix with shape ``(N, D_price)``.
    text:
        Contiguous float32 tensor with shape ``(N, K, D_text)``, where ``K``
        is the number of requested embedding families.
    mask:
        Float32 availability mask with shape ``(N, K)``. A value of one means
        that family supplied text for the row; zero means the corresponding
        text vector was filled with zeros.
    target:
        Float32 direction-label vector with shape ``(N,)``. It may contain
        NaN values only when ``require_target=False``.

    Notes
    -----
    The join with ``price_latents`` is inner because a fusion example cannot
    be constructed without a price representation. Text joins are left joins:
    missing news must not remove an otherwise valid market row.
    """
    # Use one integer type everywhere. This is especially important during the
    # final refit, where test row IDs are temporarily offset from training IDs.
    scoped_ids = row_ids.select(pl.col("row_id").cast(pl.UInt64)).unique()
    scoped_price = price_latents.with_columns(pl.col("row_id").cast(pl.UInt64))
    scoped_targets = targets.select([
        pl.col("row_id").cast(pl.UInt64), "target_up"
    ])
    frame = (
        # Keep only requested rows that have a frozen TimesFM representation.
        scoped_ids.join(scoped_price, on="row_id", how="inner")
        # Labels are attached without determining row inclusion at this stage.
        .join(scoped_targets, on="row_id", how="left")
    )
    price_columns = sorted(c for c in frame.columns if c.startswith("price_latent_"))
    text_groups, availability = [], []
    for family in families:
        family_frame = pl.read_parquet(row_text_paths[family]).with_columns(
            pl.col("row_id").cast(pl.UInt64)
        )
        family_columns = sorted(
            c for c in family_frame.columns if c.startswith(f"{family}_latent_")
        )
        if not family_columns:
            raise ValueError(f"No latent columns found for {family}")
        flag = f"has_{family}_latent"
        # Preserve rows without this family's news. Their family vector becomes
        # zero, while the separate flag tells the network that it is missing.
        frame = frame.join(
            family_frame.with_columns(pl.lit(1.0).cast(pl.Float32).alias(flag)),
            on="row_id",
            how="left",
        ).with_columns(
            pl.col(flag).fill_null(0.0),
            pl.col(family_columns).fill_null(0.0),
        )
        text_groups.append(family_columns)
        availability.append(flag)
    if require_target:
        frame = frame.filter(pl.col("target_up").is_not_null())
    # Stable chronological ordering makes saved predictions easy to audit.
    frame = frame.sort(["date", "ticker"])

    # Convert each modality to the layout expected by
    # CovariateResidualFusion.forward().
    price = np.ascontiguousarray(frame.select(price_columns).to_numpy().astype(np.float32))
    text = np.ascontiguousarray(np.stack([
        frame.select(columns).to_numpy().astype(np.float32) for columns in text_groups
    ], axis=1))
    mask = np.ascontiguousarray(frame.select(availability).to_numpy().astype(np.float32))
    target = frame["target_up"].to_numpy().astype(np.float32)
    _require_finite("TimesFM price latents", price)
    _require_finite("text-coder latents", text)
    _require_finite("text-family availability mask", mask)
    if require_target:
        _require_finite("training targets", target)
    return frame.select(["row_id", "date", "ticker", "target_up"]), price, text, mask, target


def _metric_row(
    feature_set: str,
    model_name: str,
    config_id: str,
    params_json: str,
    fold: int,
    train_rows: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    _require_finite("evaluation scores", y_score)
    y_pred = (y_score >= threshold).astype(np.int8)
    return {
        "feature_set": feature_set,
        "model": model_name,
        "config_id": config_id,
        "params_json": params_json,
        "decision_threshold": float(threshold),
        "fold": int(fold),
        "train_rows_used": int(train_rows),
        "validation_rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def _aggregate_fold_metrics(fold_metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        fold_metrics.group_by(["feature_set", "model", "config_id", "params_json"])
        .agg(
            pl.len().alias("completed_folds"),
            pl.col("accuracy").mean().alias("mean_accuracy"),
            pl.col("accuracy").std().alias("std_accuracy"),
            pl.col("balanced_accuracy").mean().alias("mean_balanced_accuracy"),
            pl.col("balanced_accuracy").std().alias("std_balanced_accuracy"),
            pl.col("roc_auc").mean().alias("mean_roc_auc"),
            pl.col("roc_auc").std().alias("std_roc_auc"),
            pl.col("brier_score").mean().alias("mean_brier_score"),
            pl.col("decision_threshold").first().alias("decision_threshold"),
        )
        .sort("mean_balanced_accuracy", descending=True)
    )


def _covariate_matrix(
    index: pl.DataFrame,
    features: pl.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    """Return numeric covariates in exactly the same row order as ``index``."""
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(f"Missing engineered covariates: {missing}")
    aligned = (
        index.select(pl.col("row_id").cast(pl.UInt64))
        .with_row_index("_order")
        .join(
            features.select([
                pl.col("row_id").cast(pl.UInt64),
                *[pl.col(column).cast(pl.Float64) for column in columns],
            ]),
            on="row_id",
            how="left",
            validate="1:1",
        )
        .sort("_order")
    )
    if aligned.select(pl.any_horizontal(pl.col(columns).is_null())).to_series().any():
        # Nulls are deliberately retained as NaN for fold-local imputation.
        aligned = aligned.with_columns(pl.col(columns).fill_null(float("nan")))
    return np.ascontiguousarray(
        aligned.select(columns).to_numpy().astype(np.float32, copy=False)
    )


def _fit_covariate_scaler(values: np.ndarray) -> dict[str, np.ndarray]:
    """Fit median imputation and standardization on training rows only."""
    clean = values.astype(np.float64, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    median = np.empty(clean.shape[1], dtype=np.float64)
    for column in range(clean.shape[1]):
        finite = clean[np.isfinite(clean[:, column]), column]
        median[column] = np.median(finite) if finite.size else 0.0
    filled = np.where(np.isfinite(clean), clean, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"median": median, "mean": mean, "scale": scale}


def _apply_covariate_scaler(
    values: np.ndarray,
    scaler: dict[str, np.ndarray],
) -> np.ndarray:
    clean = values.astype(np.float64, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    filled = np.where(np.isfinite(clean), clean, scaler["median"])
    return np.ascontiguousarray(
        ((filled - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    )


def _save_covariate_scaler(
    path: Path,
    columns: Sequence[str],
    scaler: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "covariate": list(columns),
        "imputation_median": scaler["median"],
        "standardization_mean": scaler["mean"],
        "standardization_scale": scaler["scale"],
    }).write_csv(path)


def _best_oof_threshold(predictions: pl.DataFrame) -> float:
    """Choose a probability cutoff by mean fold balanced accuracy."""
    candidates = np.linspace(0.35, 0.65, 31)
    scored: list[tuple[float, float]] = []
    for threshold in candidates:
        fold_scores = []
        for partition in predictions.partition_by("fold"):
            truth = partition["y_true"].to_numpy()
            pred = (partition["y_score"].to_numpy() >= threshold).astype(np.int8)
            fold_scores.append(balanced_accuracy_score(truth, pred))
        scored.append((float(np.mean(fold_scores)), float(threshold)))
    # On a tie, prefer the threshold closest to the conventional 0.5 cutoff.
    return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]


def run_walk_forward_fusion(
    *,
    data_dir: Path,
    output_dir: Path,
    baseline_dir: Path,
    train_price_latents: pl.DataFrame,
    test_price_latents: pl.DataFrame,
    train_features: pl.DataFrame,
    test_features: pl.DataFrame,
    train_links: pl.DataFrame,
    test_links: pl.DataFrame,
    train_targets: pl.DataFrame,
    test_targets: pl.DataFrame,
    fold_assignments: pl.DataFrame,
    requested_families: Sequence[str],
    covariate_columns: Sequence[str],
    device: str,
    latent_dim: int = 128,
    text_epochs: int = 3,
    fusion_epochs: int = 10,
    text_batch_size: int = 1024,
    fusion_batch_size: int = 512,
    fusion_hidden_dim: int = 256,
    market_depth: int = 2,
    fusion_depth: int = 2,
    residual_expansion: int = 2,
    fusion_dropout: float = 0.1,
    tuning_trials: int = 20,
    tune_hyperparameters: bool = True,
    forecast_horizon_weekdays: int = 20,
    submission_years: Sequence[int] = (2022, 2023),
    expected_submission_rows: int = 52_000,
    raw_test_path: Path | None = None,
    seed: int = 42,
    run_training: bool = True,
    force_refresh: bool = False,
) -> dict[str, pl.DataFrame]:
    """Tune on walk-forward folds, refit on train, evaluate test, and submit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_test_path = Path(raw_test_path or data_dir / "test.parquet")
    feature_set = "timesfm_covariates_original_text_residual_fusion"
    fold_numbers = fold_assignments["fold"].unique().sort().to_list()
    if not covariate_columns:
        raise ValueError("covariate_columns cannot be empty")

    catalog_rows, families = [], []
    for family in requested_families:
        path = data_dir / f"{family}_textemb.parquet"
        try:
            input_dim = parquet_embedding_dim(path)
            families.append(family)
            catalog_rows.append({
                "family": family, "usable": True,
                "input_dim": input_dim, "status": "ok",
            })
        except Exception as error:
            catalog_rows.append({
                "family": family, "usable": False,
                "input_dim": None, "status": str(error),
            })
    if not families:
        raise RuntimeError("No usable original embedding family was found")
    pl.DataFrame(catalog_rows, infer_schema_length=None).write_csv(
        output_dir / "text_family_catalog.csv"
    )
    variants = {family: (family,) for family in families}
    if len(families) > 1:
        variants["all_families"] = tuple(families)
    tuning_variant = "all_families" if "all_families" in variants else families[0]

    # Train text coders once per fold. Hyperparameter trials only retrain the
    # relatively small market/fusion network, not the expensive article coder.
    fold_scopes: dict[int, dict] = {}
    for fold in fold_numbers:
        fold_dir = output_dir / "fold_results" / f"fold_{fold}"
        train_ids = fold_assignments.filter(
            (pl.col("fold") == fold) & (pl.col("split") == "train")
        ).select("row_id")
        validation_ids = fold_assignments.filter(
            (pl.col("fold") == fold) & (pl.col("split") == "validation")
        ).select("row_id")
        fit_targets = train_targets.join(train_ids, on="row_id", how="semi")
        validation_links = train_links.join(validation_ids, on="row_id", how="semi")
        row_paths = _scope_text_latents(
            f"fold_{fold}", fold_dir, data_dir, train_links, validation_links,
            fit_targets, families, latent_dim, device, text_epochs,
            text_batch_size, force_refresh, run_training,
        )
        fold_scopes[int(fold)] = {
            "dir": fold_dir, "train_ids": train_ids,
            "validation_ids": validation_ids, "row_paths": row_paths,
        }

    default_params = {
        "hidden_dim": int(fusion_hidden_dim),
        "market_depth": int(market_depth),
        "fusion_depth": int(fusion_depth),
        "expansion": int(residual_expansion),
        "dropout": float(fusion_dropout),
        "epochs": int(fusion_epochs),
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    }

    def fold_arrays(fold: int, variant_families: Sequence[str]):
        scope = fold_scopes[int(fold)]
        train_arrays = _assemble_fusion_arrays(
            scope["train_ids"], train_price_latents, train_targets,
            scope["row_paths"], variant_families,
        )
        val_arrays = _assemble_fusion_arrays(
            scope["validation_ids"], train_price_latents, train_targets,
            scope["row_paths"], variant_families,
        )
        train_index, val_index = train_arrays[0], val_arrays[0]
        train_raw = _covariate_matrix(train_index, train_features, covariate_columns)
        val_raw = _covariate_matrix(val_index, train_features, covariate_columns)
        scaler = _fit_covariate_scaler(train_raw)
        return (
            train_arrays, val_arrays,
            _apply_covariate_scaler(train_raw, scaler),
            _apply_covariate_scaler(val_raw, scaler), scaler,
        )

    tuning_trials_table = pl.DataFrame()
    if tune_hyperparameters and tuning_trials > 0:
        try:
            import optuna
        except ImportError as error:
            raise ImportError(
                "Optuna tuning is enabled. Install it in this kernel with "
                "`%pip install optuna`, restart the kernel, and rerun."
            ) from error
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name="timesfm_covariate_residual_fusion",
            storage=f"sqlite:///{(output_dir / 'optuna_study.db').resolve()}",
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
        )

        def objective(trial) -> float:
            params = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 384]),
                "market_depth": trial.suggest_int("market_depth", 1, 3),
                "fusion_depth": trial.suggest_int("fusion_depth", 1, 3),
                "expansion": trial.suggest_categorical("expansion", [1, 2, 4]),
                "dropout": trial.suggest_float("dropout", 0.05, 0.30),
                "epochs": int(fusion_epochs),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-3, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
            }
            scores = []
            for step, fold in enumerate(fold_numbers):
                train_a, val_a, train_cov, val_cov, _ = fold_arrays(
                    int(fold), variants[tuning_variant]
                )
                model, _ = fit_fusion_model(
                    train_a[1], train_cov, train_a[2], train_a[3], train_a[4],
                    device, batch_size=fusion_batch_size,
                    seed=seed + 1000 * trial.number + int(fold), **params,
                )
                score = predict_fusion(
                    model, val_a[1], val_cov, val_a[2], val_a[3], device
                )
                scores.append(balanced_accuracy_score(
                    val_a[4].astype(np.int8), (score >= 0.5).astype(np.int8)
                ))
                trial.report(float(np.mean(scores)), step)
                del model, train_a, val_a, train_cov, val_cov
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        remaining_trials = max(0, int(tuning_trials) - len(study.trials))
        if remaining_trials:
            study.optimize(
                objective, n_trials=remaining_trials, gc_after_trial=True
            )
        best_params = {**default_params, **study.best_trial.params}
        best_params["epochs"] = int(fusion_epochs)
        trial_rows = []
        for trial in study.trials:
            trial_rows.append({
                "trial": trial.number, "state": trial.state.name,
                "mean_balanced_accuracy": trial.value,
                "params_json": json.dumps(trial.params, sort_keys=True),
            })
        tuning_trials_table = pl.DataFrame(trial_rows, infer_schema_length=None)
        tuning_trials_table.write_csv(output_dir / "optuna_trials.csv")
    else:
        best_params = default_params
    (output_dir / "best_hyperparameters.json").write_text(json.dumps({
        "selection_variant": tuning_variant,
        "selection_metric": "mean_walk_forward_balanced_accuracy",
        "test_data_used_for_tuning": False,
        "params": best_params,
    }, indent=2, sort_keys=True))

    # Refit the selected architecture within every fold and save true OOS scores.
    prediction_parts = []
    params_json = json.dumps(best_params, sort_keys=True)
    for fold in fold_numbers:
        fold = int(fold)
        for variant, variant_families in variants.items():
            train_a, val_a, train_cov, val_cov, scaler = fold_arrays(
                fold, variant_families
            )
            model_name = f"timesfm_covariates_plus_{variant}"
            config_id = f"{model_name}__optuna_best"
            model_path = fold_scopes[fold]["dir"] / "fusion_models" / f"{model_name}.pt"
            model, history = fit_fusion_model(
                train_a[1], train_cov, train_a[2], train_a[3], train_a[4],
                device, batch_size=fusion_batch_size, seed=seed + fold,
                **best_params,
            )
            save_torch_model(model_path, model, {
                "scope": f"fold_{fold}", "model": model_name,
                "price_encoder": "timesfm", "price_encoder_frozen": True,
                "covariates": list(covariate_columns),
                "covariate_preprocessing": "training_median_then_zscore",
                "text_families": list(variant_families),
                "architecture": "residual_market_mlp_then_simple_text_concatenation",
                "hyperparameters": best_params,
            })
            _save_covariate_scaler(
                fold_scopes[fold]["dir"] / "fusion_models" /
                f"{model_name}_covariate_scaler.csv",
                covariate_columns, scaler,
            )
            pl.DataFrame(history).write_csv(
                fold_scopes[fold]["dir"] / "fusion_models" /
                f"{model_name}_training_history.csv"
            )
            score = predict_fusion(
                model, val_a[1], val_cov, val_a[2], val_a[3], device
            )
            prediction_parts.append(val_a[0].with_columns(
                pl.lit(feature_set).alias("feature_set"),
                pl.lit(model_name).alias("model"),
                pl.lit(config_id).alias("config_id"),
                pl.lit(fold).cast(pl.Int8).alias("fold"),
                pl.col("target_up").cast(pl.Int8).alias("y_true"),
                pl.Series("y_score", score.astype(np.float32)),
            ).drop("target_up"))
            del model, train_a, val_a, train_cov, val_cov
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    raw_oof = pl.concat(prediction_parts)
    thresholds = {
        model: _best_oof_threshold(raw_oof.filter(pl.col("model") == model))
        for model in raw_oof["model"].unique().to_list()
    }
    threshold_table = pl.DataFrame({
        "model": list(thresholds), "decision_threshold": list(thresholds.values())
    })
    threshold_table.write_csv(output_dir / "selected_decision_thresholds.csv")
    oof_predictions = (
        raw_oof.join(threshold_table, on="model", how="left")
        .with_columns(
            (pl.col("y_score") >= pl.col("decision_threshold"))
            .cast(pl.Int8).alias("y_pred")
        )
        .select([
            "feature_set", "model", "config_id", "fold", "row_id", "date",
            "ticker", "y_true", "y_score", "y_pred", "decision_threshold",
        ])
        .sort(["model", "fold", "date", "ticker"])
    )
    metric_rows = []
    for partition in oof_predictions.partition_by(["model", "fold"], as_dict=False):
        first = partition.row(0, named=True)
        train_rows = fold_scopes[int(first["fold"])]["train_ids"].height
        metric_rows.append(_metric_row(
            feature_set, first["model"], first["config_id"], params_json,
            int(first["fold"]), train_rows, partition["y_true"].to_numpy(),
            partition["y_score"].to_numpy(), float(first["decision_threshold"]),
        ))
    fold_metrics = pl.DataFrame(metric_rows).sort(["model", "fold"])
    aggregate = _aggregate_fold_metrics(fold_metrics)
    for fold in fold_numbers:
        fold_metrics.filter(pl.col("fold") == fold).write_csv(
            fold_scopes[int(fold)]["dir"] / f"selected_model_metrics_{feature_set}.csv"
        )
    fold_metrics.write_csv(output_dir / f"selected_config_fold_metrics_{feature_set}.csv")
    aggregate.write_csv(output_dir / f"aggregate_model_selection_summary_{feature_set}.csv")
    oof_predictions.write_parquet(
        output_dir / f"selected_config_oof_predictions_{feature_set}.parquet",
        compression="zstd",
    )

    # Fit fresh coders, scaler, and networks on every labeled training row.
    final_dir = output_dir / "final_refit"
    final_targets = train_targets.filter(pl.col("target_up").is_not_null())
    test_row_offset = int(train_targets["row_id"].max()) + 1
    offset = pl.lit(test_row_offset, dtype=pl.UInt64)
    scoped_test_links = test_links.with_columns(
        (pl.col("row_id").cast(pl.UInt64) + offset).alias("row_id")
    )
    scoped_test_price = test_price_latents.with_columns(
        (pl.col("row_id").cast(pl.UInt64) + offset).alias("row_id")
    )
    scoped_test_targets = test_targets.with_columns(
        (pl.col("row_id").cast(pl.UInt64) + offset).alias("row_id")
    )
    scoped_test_features = test_features.with_columns(
        (pl.col("row_id").cast(pl.UInt64) + offset).alias("row_id")
    )
    final_row_paths = _scope_text_latents(
        "final", final_dir, data_dir, train_links, scoped_test_links,
        final_targets, families, latent_dim, device, text_epochs,
        text_batch_size, force_refresh, run_training,
    )
    all_train_ids = final_targets.select("row_id")
    all_test_ids = scoped_test_price.select("row_id")
    final_prediction_parts = []
    for variant, variant_families in variants.items():
        train_a = _assemble_fusion_arrays(
            all_train_ids, train_price_latents, train_targets,
            final_row_paths, variant_families,
        )
        test_a = _assemble_fusion_arrays(
            all_test_ids, scoped_test_price, scoped_test_targets,
            final_row_paths, variant_families, require_target=False,
        )
        train_raw = _covariate_matrix(train_a[0], train_features, covariate_columns)
        test_raw = _covariate_matrix(test_a[0], scoped_test_features, covariate_columns)
        scaler = _fit_covariate_scaler(train_raw)
        train_cov = _apply_covariate_scaler(train_raw, scaler)
        test_cov = _apply_covariate_scaler(test_raw, scaler)
        model_name = f"timesfm_covariates_plus_{variant}"
        model, history = fit_fusion_model(
            train_a[1], train_cov, train_a[2], train_a[3], train_a[4],
            device, batch_size=fusion_batch_size, seed=seed, **best_params,
        )
        save_torch_model(final_dir / "fusion_models" / f"{model_name}.pt", model, {
            "scope": "final", "model": model_name,
            "price_encoder": "timesfm", "price_encoder_frozen": True,
            "covariates": list(covariate_columns),
            "covariate_preprocessing": "all-training median then zscore",
            "text_families": list(variant_families),
            "architecture": "residual_market_mlp_then_simple_text_concatenation",
            "hyperparameters": best_params,
        })
        _save_covariate_scaler(
            final_dir / "fusion_models" / f"{model_name}_covariate_scaler.csv",
            covariate_columns, scaler,
        )
        pl.DataFrame(history).write_csv(
            final_dir / "fusion_models" / f"{model_name}_training_history.csv"
        )
        score = predict_fusion(
            model, test_a[1], test_cov, test_a[2], test_a[3], device
        )
        threshold = thresholds[model_name]
        final_prediction_parts.append(test_a[0].with_columns(
            (pl.col("row_id") - test_row_offset).cast(pl.UInt32).alias("row_id"),
            pl.lit(model_name).alias("model"),
            pl.Series("y_score", score.astype(np.float32)),
            pl.Series("y_pred", (score >= threshold).astype(np.int8)),
            pl.lit(threshold).alias("decision_threshold"),
            pl.col("target_up").cast(pl.Int8).alias("y_true"),
        ).drop("target_up"))
        del model, train_a, test_a, train_raw, test_raw, train_cov, test_cov
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    final_predictions = (
        pl.concat(final_prediction_parts)
        .with_columns(
            pl.col("date").dt.add_business_days(forecast_horizon_weekdays)
            .alias("prediction_date")
        )
        .with_columns(pl.col("prediction_date").dt.year().alias("test_year"))
        .sort(["model", "prediction_date", "ticker"])
    )
    final_predictions.write_parquet(
        output_dir / "final_test_predictions.parquet", compression="zstd"
    )

    # Report held-out test performance separately for target years 2022/2023.
    test_metric_rows = []
    for model_name in final_predictions["model"].unique().sort().to_list():
        for year in submission_years:
            part = final_predictions.filter(
                (pl.col("model") == model_name) & (pl.col("test_year") == year)
            )
            scored = part.filter(pl.col("y_true").is_not_null())
            row = {
                "feature_set": feature_set, "model": model_name,
                "test_year": int(year), "total_prediction_rows": part.height,
                "scored_rows": scored.height,
                "coverage": scored.height / part.height if part.height else 0.0,
                "decision_threshold": thresholds[model_name],
            }
            if scored.height:
                row.update(directional_classification_metrics(
                    scored["y_true"].to_numpy(), scored["y_score"].to_numpy(),
                    thresholds[model_name],
                ))
            test_metric_rows.append(row)
    final_metrics = pl.DataFrame(test_metric_rows, infer_schema_length=None)
    final_metrics.write_csv(output_dir / "final_test_metrics_by_year.csv")

    # Convert directional probabilities to positive Close predictions and save
    # one organizer-format submission per model, always containing both years.
    return_column = "fwd_log_return_20"
    if return_column not in train_targets.columns:
        raise ValueError(f"{return_column} is needed for training-only price scaling")
    return_scale = train_targets.select(
        pl.col(return_column).drop_nulls().abs().median()
    ).item()
    if return_scale is None or not np.isfinite(return_scale) or return_scale <= 0:
        raise ValueError("Could not derive a positive submission return scale")
    raw_test_close = (
        pl.scan_parquet(raw_test_path)
        .select(
            pl.col("date").cast(pl.Date), "ticker",
            pl.col("close").cast(pl.Float64).alias("origin_close"),
        )
        .collect()
        .unique(["date", "ticker"])
    )
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    expected_per_year = expected_submission_rows // len(submission_years)
    for model_name in final_predictions["model"].unique().sort().to_list():
        source = (
            final_predictions.filter(
                (pl.col("model") == model_name)
                & pl.col("test_year").is_in(submission_years)
            )
            .join(raw_test_close, on=["date", "ticker"], how="left", validate="m:1")
        )
        close = probability_to_price(
            source["origin_close"].to_numpy(), source["y_score"].to_numpy(),
            float(return_scale), source["decision_threshold"].to_numpy(),
        )
        submission = source.select([
            pl.concat_str([
                pl.col("ticker"), pl.lit("_"),
                pl.col("prediction_date").dt.strftime("%Y-%m-%d"),
            ]).alias("ID")
        ]).with_columns(pl.Series("Close", close)).sort("ID")
        year_counts = source.group_by("test_year").len()
        counts = dict(year_counts.iter_rows())
        if any(counts.get(int(year), 0) != expected_per_year for year in submission_years):
            raise ValueError(
                f"{model_name} does not have {expected_per_year:,} predictions per year: {counts}"
            )
        if submission.height != expected_submission_rows:
            raise ValueError(
                f"{model_name} submission has {submission.height:,} rows; "
                f"expected {expected_submission_rows:,}"
            )
        if submission["ID"].n_unique() != expected_submission_rows:
            raise ValueError(f"{model_name} submission IDs are not unique")
        if not np.isfinite(submission["Close"].to_numpy()).all() or (
            submission["Close"] <= 0
        ).any():
            raise ValueError(f"{model_name} submission contains an invalid Close")
        path = submission_dir / f"{model_name}_submission.csv"
        submission.write_csv(path)
        manifest_rows.append({
            "model": model_name, "path": str(path), "rows": submission.height,
            "years": ",".join(map(str, submission_years)),
            "return_scale": float(return_scale),
            "decision_threshold": thresholds[model_name],
        })
    submission_manifest = pl.DataFrame(manifest_rows, infer_schema_length=None)
    submission_manifest.write_csv(output_dir / "submission_manifest.csv")

    baseline_tables = [
        pl.read_csv(path)
        for path in sorted(baseline_dir.glob("*/selected_config_fold_metrics_*.csv"))
    ]
    comparison_folds = pl.concat(
        [*baseline_tables, fold_metrics], how="diagonal_relaxed"
    ) if baseline_tables else fold_metrics
    comparison_aggregate = _aggregate_fold_metrics(
        comparison_folds.select(fold_metrics.columns)
    )
    comparison_folds.write_csv(output_dir / "comparison_with_baselines_fold_metrics.csv")
    comparison_aggregate.write_csv(output_dir / "comparison_with_baselines_aggregate.csv")
    return {
        "tuning_trials": tuning_trials_table,
        "fold_metrics": fold_metrics, "aggregate": aggregate,
        "oof_predictions": oof_predictions,
        "final_predictions": final_predictions, "final_metrics": final_metrics,
        "submission_manifest": submission_manifest,
        "comparison_folds": comparison_folds,
        "comparison_aggregate": comparison_aggregate,
    }
