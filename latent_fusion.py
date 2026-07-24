"""Memory-conscious helpers for frozen-price, covariate, and text fusion.

The notebook owns TimesFM loading and hidden-state extraction. This module
streams raw embedding families through trainable in-model adapters, preserves
semantic text-field tokens, applies shared text self-attention, and uses the
market representation as a cross-attention query before fusion.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from pathlib import Path
from typing import Sequence
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_curve,
    roc_auc_score,
)
from utils import directional_classification_metrics, probability_to_price


def parquet_embedding_dim(path: Path) -> int:
    """Return the number of emb_* columns and validate the parquet footer."""
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception as error:
        raise ValueError(
            f"{path} is not a readable Parquet embedding file; "
            "it may be incomplete or corrupted"
        ) from error
    columns = [name for name in schema.names if name.startswith("emb_")]
    if "text_id" not in schema.names or not columns:
        raise ValueError(f"{path} must contain text_id and emb_* columns")
    return len(columns)


class RawEmbeddingStore:
    """Disk-backed raw embedding matrix and its text-ID row index."""

    def __init__(
        self,
        family: str,
        values_path: Path,
        index_path: Path,
        input_dim: int,
    ):
        self.family = family
        self.input_dim = int(input_dim)
        self.values = np.load(values_path, mmap_mode="r")
        if self.values.ndim != 2 or self.values.shape[1] != self.input_dim:
            raise ValueError(f"Invalid raw embedding cache for {family}")
        self.index = pl.read_parquet(index_path).select([
            "text_id", pl.col("embedding_row").cast(pl.Int64),
        ])

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """Load only one mini-batch of raw vectors; -1 denotes missing text."""
        safe = np.maximum(indices, 0)
        values = np.asarray(self.values[safe], dtype=np.float32)
        values[indices < 0] = 0.0
        return np.ascontiguousarray(values)


def prepare_raw_embedding_store(
    *,
    family: str,
    embedding_path: Path,
    cache_dir: Path,
    batch_size: int = 2048,
) -> RawEmbeddingStore:
    """Stream a family parquet once into a float32 random-access matrix."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    values_path = cache_dir / f"{family}_raw_embeddings_float32.npy"
    index_path = cache_dir / f"{family}_text_id_index.parquet"
    metadata_path = cache_dir / f"{family}_raw_embeddings_metadata.json"
    input_dim = parquet_embedding_dim(embedding_path)
    parquet_file = pq.ParquetFile(embedding_path)
    row_count = parquet_file.metadata.num_rows
    source_stat = embedding_path.stat()
    expected = {
        "family": family,
        "source": str(embedding_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "rows": row_count,
        "input_dim": input_dim,
        "dtype": "float32",
    }
    valid = False
    if values_path.exists() and index_path.exists() and metadata_path.exists():
        try:
            valid = json.loads(metadata_path.read_text()) == expected
            if valid:
                cached = np.load(values_path, mmap_mode="r")
                valid = cached.shape == (row_count, input_dim)
                del cached
        except Exception:
            valid = False
    if not valid:
        required_bytes = row_count * input_dim * np.dtype(np.float32).itemsize
        free_bytes = shutil.disk_usage(cache_dir).free
        if free_bytes < int(required_bytes * 1.05):
            raise OSError(
                f"Insufficient disk space for the {family} raw embedding cache: "
                f"need about {required_bytes / 2**30:.2f} GiB plus overhead, "
                f"but only {free_bytes / 2**30:.2f} GiB is free"
            )
        gib = row_count * input_dim * np.dtype(np.float32).itemsize / 2**30
        print(
            f"Building raw {family} embedding cache: "
            f"{row_count:,} x {input_dim:,} float32 ({gib:.2f} GiB)"
        )
        temporary_values = cache_dir / f"{family}_raw_embeddings_building.npy"
        temporary_index = cache_dir / f"{family}_text_id_index_building.parquet"
        matrix = np.lib.format.open_memmap(
            temporary_values,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, input_dim),
        )
        text_ids: list[str] = []
        columns = ["text_id"] + [f"emb_{index}" for index in range(input_dim)]
        start = 0
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=columns
        ):
            stop = start + record_batch.num_rows
            text_ids.extend(record_batch.column(0).to_pylist())
            matrix[start:stop] = np.column_stack([
                record_batch.column(column).to_numpy(zero_copy_only=False)
                for column in range(1, input_dim + 1)
            ]).astype(np.float32, copy=False)
            start = stop
            if start == row_count or start % (batch_size * 50) == 0:
                print(
                    f"  {family}: cached {start:,}/{row_count:,} embeddings"
                )
        if start != row_count:
            raise RuntimeError(
                f"Raw embedding cache for {family} wrote {start} of {row_count} rows"
            )
        matrix.flush()
        del matrix
        pl.DataFrame({
            "text_id": text_ids,
            "embedding_row": np.arange(row_count, dtype=np.int64),
        }).write_parquet(temporary_index, compression="zstd")
        temporary_values.replace(values_path)
        temporary_index.replace(index_path)
        metadata_path.write_text(json.dumps(expected, indent=2, sort_keys=True))
    else:
        print(f"Using raw {family} embedding cache: {values_path}")
    return RawEmbeddingStore(family, values_path, index_path, input_dim)


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


# Legacy projected-token baseline components are retained for old checkpoint
# compatibility. The active walk-forward path uses RawText*Fusion below and
# never trains or invokes TextCoder.
# Inline shape symbols below: B=batch, T=time, K=text families, V=temporal
# variables, H=hidden width, and D_*=the named feature/latent dimension.
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
        # values:     (B, D_text_in)
        projected = self.projection(values)       # (B, D_text_latent)
        activated = F.gelu(projected)             # (B, D_text_latent)
        dropped = self.dropout(activated)         # (B, D_text_latent)
        return self.norm(dropped)                 # (B, D_text_latent)


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
        # values:     (B, H)
        normalized = self.norm(values)            # (B, H)
        hidden = self.mlp[0](normalized)           # (B, H * expansion)
        hidden = self.mlp[1](hidden)               # (B, H * expansion)
        hidden = self.mlp[2](hidden)               # (B, H * expansion)
        correction = self.mlp[3](hidden)           # (B, H)
        correction = self.mlp[4](correction)       # (B, H)
        return values + correction                 # (B, H)


class MarketConditionedArticlePool(nn.Module):
    """Pool field-preserving article tokens using the current market state."""

    def __init__(
        self,
        text_dim: int,
        market_dim: int,
        family_count: int,
        field_count: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.family_count = int(family_count)
        self.field_count = int(field_count)
        self.text_dim = int(text_dim)
        self.family_embedding = nn.Parameter(
            torch.empty(1, family_count, 1, text_dim)
        )
        self.field_embedding = nn.Parameter(
            torch.empty(1, 1, field_count, text_dim)
        )
        nn.init.normal_(self.family_embedding, std=0.02)
        nn.init.normal_(self.field_embedding, std=0.02)
        self.article_grn = GatedResidualNetwork(
            text_dim, text_dim, text_dim, dropout=dropout
        )
        self.market_query = nn.Linear(market_dim, text_dim)
        self.output_grn = GatedResidualNetwork(
            text_dim,
            text_dim,
            text_dim,
            context_dim=market_dim,
            dropout=dropout,
        )

    def forward(
        self,
        articles: torch.Tensor,
        article_mask: torch.Tensor,
        market: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # articles: (B, K, A, D_text); article_mask: (B, K, A)
        # market: (B, H)
        identified = (
            articles + self.family_embedding + self.field_embedding
        )                                             # (B, K, A, D_text)
        tokens = self.article_grn(identified)         # (B, K, A, D_text)
        batch_size = tokens.shape[0]
        token_count = self.family_count * self.field_count
        tokens = tokens.reshape(
            batch_size, token_count, self.text_dim
        )                                             # (B, K * A, D_text)
        flat_mask = article_mask.reshape(
            batch_size, token_count
        )                                             # (B, K * A)
        query = self.market_query(market)             # (B, D_text)
        scores = torch.einsum(
            "bqd,bd->bq", tokens, query
        ) / self.text_dim ** 0.5                      # (B, K * A)
        scores = scores.masked_fill(
            flat_mask <= 0,
            torch.finfo(scores.dtype).min,
        )                                             # (B, K * A)
        weights = torch.softmax(scores, dim=1)        # (B, K * A)
        weights = weights * flat_mask                 # (B, K * A)
        weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)                             # (B, K * A)
        pooled = torch.einsum(
            "bq,bqd->bd", weights, tokens
        )                                             # (B, D_text)
        pooled = self.output_grn(pooled, market)      # (B, D_text)
        has_text = flat_mask.any(dim=1, keepdim=True) # (B, 1)
        pooled = pooled * has_text.to(pooled.dtype)   # (B, D_text)
        return pooled, weights                        # (B, D_text), (B, K * A)


class CovariateResidualFusion(nn.Module):
    """Fuse a residual market MLP with family-specific text representations.

    The frozen TimesFM vector is concatenated with fold-standardized engineered
    covariates first.  A residual MLP turns that joint numeric input into the
    market representation. Original text-embedding families and semantic text
    fields remain separate until market-conditioned masked attention pools the
    available article tokens for the residual fusion head.
    """

    def __init__(
        self,
        price_dim: int,
        covariate_dim: int,
        text_dim: int,
        family_count: int,
        field_count: int,
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
        self.field_count = int(field_count)
        self.hidden_dim = int(hidden_dim)
        self.market_depth = int(market_depth)
        self.fusion_depth = int(fusion_depth)
        self.expansion = int(expansion)
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
        self.text_pool = MarketConditionedArticlePool(
            text_dim,
            hidden_dim,
            family_count,
            field_count,
            dropout,
        )
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
        # price: (B, D_price); covariates: (B, D_cov)
        # family_text: (B, K, A, D_text); family_mask: (B, K, A)
        market_input = torch.cat(
            [price, covariates], dim=1
        )                                           # (B, D_price + D_cov)
        market = self.market_input[0](market_input) # (B, D_price + D_cov)
        market = self.market_input[1](market)       # (B, H)
        market = self.market_input[2](market)       # (B, H)
        market = self.market_input[3](market)       # (B, H)
        for block in self.market_blocks:
            market = block(market)                  # (B, H)
        market = self.market_norm(market)           # (B, H)
        h_text, _ = self.text_pool(
            family_text, family_mask, market
        )                                           # (B, D_text)
        fusion_input = torch.cat(
            [market, h_text], dim=1
        )                                           # (B, H + D_text)
        fused = self.fuse_input[0](fusion_input)    # (B, H)
        fused = self.fuse_input[1](fused)           # (B, H)
        fused = self.fuse_input[2](fused)           # (B, H)
        for block in self.residual_blocks:
            fused = block(fused)                    # (B, H)
        fused = self.output_norm(fused)             # (B, H)
        logits = self.classifier(fused)             # (B, 1)
        return logits.squeeze(1)                    # (B,)


class GatedResidualNetwork(nn.Module):
    """TFT gated residual network with optional static context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.context_layer = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        )
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim * 2)
        self.skip = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, values: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        # values: (..., D_in); context: (..., D_context), if configured
        hidden = self.input_layer(values)           # (..., H)
        if self.context_layer is not None:
            if context is None:
                raise ValueError("This gated residual network requires context")
            context_hidden = self.context_layer(context)  # (..., H)
            hidden = hidden + context_hidden         # (..., H)
        hidden = F.elu(hidden)                       # (..., H)
        hidden = self.hidden_layer(hidden)           # (..., H)
        hidden = self.dropout(hidden)                # (..., H)
        gate_input = self.gate(hidden)               # (..., 2 * D_out)
        gated = F.glu(gate_input, dim=-1)            # (..., D_out)
        residual = self.skip(values)                 # (..., D_out)
        return self.norm(residual + gated)           # (..., D_out)


class UnifiedRawTextAttention(nn.Module):
    """Joint raw-family adapters, text self-attention, and market cross-attention."""

    def __init__(
        self,
        family_dims: dict[str, int],
        field_count: int,
        text_dim: int,
        market_dim: int,
        attention_heads: int = 4,
        self_attention_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if text_dim % attention_heads:
            raise ValueError("text_dim must be divisible by attention_heads")
        if self_attention_layers < 1:
            raise ValueError("self_attention_layers must be positive")
        self.families = tuple(family_dims)
        self.field_count = int(field_count)
        self.text_dim = int(text_dim)
        self.adapters = nn.ModuleDict({
            family: nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, text_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for family, input_dim in family_dims.items()
        })
        self.family_embedding = nn.Parameter(
            torch.empty(1, len(self.families), 1, text_dim)
        )
        self.field_embedding = nn.Parameter(
            torch.empty(1, 1, field_count, text_dim)
        )
        nn.init.normal_(self.family_embedding, std=0.02)
        nn.init.normal_(self.field_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=attention_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self_attention_layers,
            enable_nested_tensor=False,
        )
        self.market_query = nn.Linear(market_dim, text_dim)
        self.cross_attention = nn.MultiheadAttention(
            text_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(text_dim)
        self.output_grn = GatedResidualNetwork(
            text_dim,
            text_dim,
            text_dim,
            context_dim=market_dim,
            dropout=dropout,
        )

    def forward(
        self,
        raw_articles: dict[str, torch.Tensor],
        article_masks: dict[str, torch.Tensor],
        market: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Each raw family: (B, A, D_family); each mask: (B, A)
        family_tokens, family_masks = [], []
        for family_index, family in enumerate(self.families):
            content = self.adapters[family](
                raw_articles[family]
            )                                         # (B, A, D_text)
            identified = (
                content
                + self.family_embedding[:, family_index]
                + self.field_embedding[:, 0]
            )                                         # (B, A, D_text)
            family_tokens.append(identified)
            family_masks.append(article_masks[family])
        tokens = torch.stack(family_tokens, dim=1)    # (B, K, A, D_text)
        available = torch.stack(family_masks, dim=1)  # (B, K, A)
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(
            batch_size, len(self.families) * self.field_count, self.text_dim
        )                                             # (B, K * A, D_text)
        available = available.reshape(
            batch_size, len(self.families) * self.field_count
        )                                             # (B, K * A)
        available = available.bool()                 # (B, K * A)
        if not available.any(dim=1).all():
            missing_rows = int((~available.any(dim=1)).sum().item())
            raise ValueError(
                f"{missing_rows} rows have no available raw text embedding"
            )
        contextual = self.self_attention(
            tokens, src_key_padding_mask=~available
        )                                             # (B, K * A, D_text)
        query = self.market_query(market).unsqueeze(1)  # (B, 1, D_text)
        attended, weights = self.cross_attention(
            query=query,
            key=contextual,
            value=contextual,
            key_padding_mask=~available,
            need_weights=True,
            average_attn_weights=True,
        )                                             # (B, 1, D_text), (B, 1, K*A)
        attended = self.cross_norm(query + attended)  # (B, 1, D_text)
        context = self.output_grn(
            attended.squeeze(1), market
        )                                             # (B, D_text)
        return context, weights.squeeze(1)            # (B, D_text), (B, K*A)


class TFTVariableSelection(nn.Module):
    """Select time-varying numeric variables using TFT-style soft weights."""

    def __init__(
        self, variable_count: int, hidden_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.variable_count = int(variable_count)
        self.variable_encoders = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(variable_count)
        ])
        self.weight_network = GatedResidualNetwork(
            variable_count,
            hidden_dim,
            variable_count,
            context_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self, values: torch.Tensor, static_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[-1] != self.variable_count:
            raise ValueError(
                "TFT temporal inputs must have shape "
                f"(batch, time, {self.variable_count})"
            )
        # values: (B, T, V); static_context: (B, H)
        context = static_context.unsqueeze(1)        # (B, 1, H)
        context = context.expand(-1, values.shape[1], -1)  # (B, T, H)
        weight_logits = self.weight_network(
            values, context
        )                                             # (B, T, V)
        weights = torch.softmax(weight_logits, dim=-1)  # (B, T, V)
        encoded = torch.stack([
            encoder(values[..., index:index + 1])    # each: (B, T, H)
            for index, encoder in enumerate(self.variable_encoders)
        ], dim=-2)                                    # (B, T, V, H)
        expanded_weights = weights.unsqueeze(-1)     # (B, T, V, 1)
        weighted = encoded * expanded_weights        # (B, T, V, H)
        selected = weighted.sum(dim=-2)              # (B, T, H)
        return selected, weights                     # (B, T, H), (B, T, V)


class TFTMarketEncoder(nn.Module):
    """Encoder-only Temporal Fusion Transformer for historical market windows."""

    def __init__(
        self,
        price_dim: int,
        covariate_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("TFT hidden_dim must be divisible by attention_heads")
        self.static_context = GatedResidualNetwork(
            price_dim + covariate_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        self.variable_selection = TFTVariableSelection(
            temporal_dim, hidden_dim, dropout
        )
        self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.output_grn = GatedResidualNetwork(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        price: torch.Tensor,
        covariates: torch.Tensor,
        temporal: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # price: (B, D_price); covariates: (B, D_cov)
        # temporal: (B, T, V); padding_mask: (B, T), True means padding
        static_input = torch.cat(
            [price, covariates], dim=-1
        )                                             # (B, D_price + D_cov)
        context = self.static_context(static_input)  # (B, H)
        selected, _ = self.variable_selection(
            temporal, context
        )                                             # selected: (B, T, H)
        recurrent, _ = self.temporal_lstm(selected)  # (B, T, H)
        lstm_gate_input = self.lstm_gate(recurrent)   # (B, T, 2H)
        gated_recurrent = F.glu(
            lstm_gate_input, dim=-1
        )                                             # (B, T, H)
        recurrent = self.lstm_norm(
            selected + gated_recurrent
        )                                             # (B, T, H)
        attended, _ = self.attention(
            recurrent,
            recurrent,
            recurrent,
            key_padding_mask=padding_mask,
            need_weights=False,
        )                                             # (B, T, H)
        attention_gate_input = self.attention_gate(
            attended
        )                                             # (B, T, 2H)
        gated_attention = F.glu(
            attention_gate_input, dim=-1
        )                                             # (B, T, H)
        attended = self.attention_norm(
            recurrent + gated_attention
        )                                             # (B, T, H)
        # Windows are left padded, so the final position is the forecast origin.
        final_timestep = attended[:, -1]             # (B, H)
        return self.output_grn(final_timestep, context)  # (B, H)


class CovariateTFTFusion(nn.Module):
    """TFT market encoder followed by the existing residual text-fusion head."""

    def __init__(
        self,
        price_dim: int,
        covariate_dim: int,
        temporal_dim: int,
        text_dim: int,
        family_count: int,
        field_count: int,
        hidden_dim: int,
        fusion_depth: int = 2,
        expansion: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.market_encoder = TFTMarketEncoder(
            price_dim,
            covariate_dim,
            temporal_dim,
            hidden_dim,
            attention_heads,
            dropout,
        )
        self.text_pool = MarketConditionedArticlePool(
            text_dim,
            hidden_dim,
            family_count,
            field_count,
            dropout,
        )
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
        temporal: torch.Tensor,
        temporal_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # price: (B, D_price); covariates: (B, D_cov)
        # family_text: (B, K, A, D_text); family_mask: (B, K, A)
        # temporal: (B, T, V); temporal_padding_mask: (B, T)
        market = self.market_encoder(
            price, covariates, temporal, temporal_padding_mask
        )                                             # (B, H)
        h_text, _ = self.text_pool(
            family_text, family_mask, market
        )                                             # (B, D_text)
        fusion_input = torch.cat(
            [market, h_text], dim=1
        )                                             # (B, H + D_text)
        fused = self.fuse_input[0](fusion_input)      # (B, H)
        fused = self.fuse_input[1](fused)             # (B, H)
        fused = self.fuse_input[2](fused)             # (B, H)
        for block in self.residual_blocks:
            fused = block(fused)                      # (B, H)
        fused = self.output_norm(fused)               # (B, H)
        logits = self.classifier(fused)               # (B, 1)
        return logits.squeeze(1)                      # (B,)


class RawTextResidualFusion(nn.Module):
    """Residual market encoder with unified end-to-end raw-text attention."""

    def __init__(
        self,
        price_dim: int,
        covariate_dim: int,
        family_dims: dict[str, int],
        field_count: int,
        text_dim: int,
        hidden_dim: int,
        market_depth: int = 2,
        fusion_depth: int = 2,
        expansion: int = 2,
        text_attention_heads: int = 4,
        text_attention_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
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

    def forward(
        self,
        price: torch.Tensor,
        covariates: torch.Tensor,
        raw_articles: dict[str, torch.Tensor],
        article_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        market = self.market_input(
            torch.cat([price, covariates], dim=1)
        )                                             # (B, H)
        for block in self.market_blocks:
            market = block(market)                    # (B, H)
        market = self.market_norm(market)             # (B, H)
        text, _ = self.text_attention(
            raw_articles, article_masks, market
        )                                             # (B, D_text)
        fused = self.fuse_input(
            torch.cat([market, text], dim=1)
        )                                             # (B, H)
        for block in self.fusion_blocks:
            fused = block(fused)                      # (B, H)
        return self.classifier(
            self.output_norm(fused)
        ).squeeze(1)                                  # (B,)


class CrossStockAttention(nn.Module):
    """Attend across stocks independently within each forecast date."""

    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError(
                "Cross-stock hidden_dim must be divisible by attention_heads"
            )
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        stock_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        # values: (B, H); stock_group_ids: (B,), equal IDs share one date.
        if stock_group_ids.ndim != 1 or len(stock_group_ids) != len(values):
            raise ValueError(
                "stock_group_ids must have shape (batch_size,)"
            )
        unique_groups, counts = torch.unique_consecutive(
            stock_group_ids, return_counts=True
        )
        if len(unique_groups) != int(torch.unique(stock_group_ids).numel()):
            raise ValueError(
                "Rows for each stock-attention date must be contiguous"
            )
        lengths = counts.tolist()
        groups = torch.split(values, lengths, dim=0)
        padded = pad_sequence(groups, batch_first=True)  # (G, S_max, H)
        positions = torch.arange(
            padded.shape[1], device=values.device
        ).unsqueeze(0)                                  # (1, S_max)
        padding_mask = positions >= counts.unsqueeze(1)  # (G, S_max)
        attended, _ = self.attention(
            padded,
            padded,
            padded,
            key_padding_mask=padding_mask,
            need_weights=False,
        )                                               # (G, S_max, H)
        attended = self.norm(
            padded + self.dropout(attended)
        )                                               # (G, S_max, H)
        return torch.cat([
            attended[index, :length]
            for index, length in enumerate(lengths)
        ], dim=0)                                       # (B, H)


class RawTextTFTFusion(nn.Module):
    """TFT market encoder with unified end-to-end raw-text attention."""

    def __init__(
        self,
        price_dim: int,
        covariate_dim: int,
        temporal_dim: int,
        family_dims: dict[str, int],
        field_count: int,
        text_dim: int,
        hidden_dim: int,
        fusion_depth: int = 2,
        expansion: int = 2,
        market_attention_heads: int = 4,
        text_attention_heads: int = 4,
        text_attention_layers: int = 1,
        cross_stock_attention: bool = False,
        cross_stock_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.market_encoder = TFTMarketEncoder(
            price_dim,
            covariate_dim,
            temporal_dim,
            hidden_dim,
            market_attention_heads,
            dropout,
        )
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
        self.cross_stock_attention = (
            CrossStockAttention(
                hidden_dim,
                cross_stock_attention_heads,
                dropout,
            )
            if cross_stock_attention else None
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        price: torch.Tensor,
        covariates: torch.Tensor,
        raw_articles: dict[str, torch.Tensor],
        article_masks: dict[str, torch.Tensor],
        temporal: torch.Tensor,
        temporal_padding_mask: torch.Tensor,
        stock_group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        market = self.market_encoder(
            price, covariates, temporal, temporal_padding_mask
        )                                             # (B, H)
        text, _ = self.text_attention(
            raw_articles, article_masks, market
        )                                             # (B, D_text)
        fused = self.fuse_input(
            torch.cat([market, text], dim=1)
        )                                             # (B, H)
        for block in self.fusion_blocks:
            fused = block(fused)                      # (B, H)
        fused = self.output_norm(fused)                # (B, H)
        if self.cross_stock_attention is not None:
            if stock_group_ids is None:
                raise ValueError(
                    "Cross-stock attention requires stock_group_ids"
                )
            fused = self.cross_stock_attention(
                fused, stock_group_ids
            )                                         # (B, H)
        return self.classifier(fused).squeeze(1)      # (B,)


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
    """Preserve one token per row/text field for learned attention pooling.

    Duplicate articles assigned to the same text field are averaged, but the
    distinct semantic fields remain separate.  The fusion network subsequently
    learns fold-local, market-conditioned weights over these field tokens.
    """
    latent_columns = [
        name for name in pq.ParquetFile(projected_path).schema_arrow.names
        if name.startswith("latent_")
    ]
    pooled = (
        links.lazy().select(["row_id", "text_field", "text_id"]).unique()
        .join(pl.scan_parquet(projected_path), on="text_id", how="inner")
        .group_by(["row_id", "text_field"])
        .agg([pl.col(name).mean().cast(pl.Float32) for name in latent_columns])
        .rename({name: f"{family}_{name}" for name in latent_columns})
        .collect(engine="streaming")
        .sort(["row_id", "text_field"])
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
    market_encoder: str = "mlp",
    market_sequence: np.ndarray | None = None,
    sequence_padding_mask: np.ndarray | None = None,
    attention_heads: int = 4,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fit the fusion map and classifier; pretrained/text encoders stay frozen."""
    _require_finite("fusion-training price latents", price)
    _require_finite("fusion-training covariates", covariates)
    _require_finite("fusion-training text latents", family_text)
    _require_finite("fusion-training family mask", family_mask)
    _require_finite("fusion-training targets", target)
    if family_text.ndim != 4 or family_mask.shape != family_text.shape[:3]:
        raise ValueError(
            "Article tokens/mask must have shapes (N, K, A, D) and (N, K, A)"
        )
    if market_encoder not in {"mlp", "tft"}:
        raise ValueError("market_encoder must be 'mlp' or 'tft'")
    if market_encoder == "tft":
        if market_sequence is None or sequence_padding_mask is None:
            raise ValueError("TFT training requires temporal sequences and masks")
        _require_finite("TFT training sequences", market_sequence)
        if market_sequence.ndim != 3 or sequence_padding_mask.shape != market_sequence.shape[:2]:
            raise ValueError("TFT sequence/mask shapes are inconsistent")
    torch.manual_seed(seed)
    if market_encoder == "tft":
        model = CovariateTFTFusion(
            price.shape[1], covariates.shape[1], market_sequence.shape[2],
            family_text.shape[3], family_text.shape[1], family_text.shape[2],
            hidden_dim,
            fusion_depth=fusion_depth, expansion=expansion,
            attention_heads=attention_heads, dropout=dropout,
        ).to(device)
    else:
        model = CovariateResidualFusion(
            price.shape[1], covariates.shape[1], family_text.shape[3],
            family_text.shape[1], family_text.shape[2], hidden_dim,
            market_depth=market_depth,
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
            t = torch.from_numpy(family_text[indices]).to(
                device=device, dtype=torch.float32
            )
            m = torch.from_numpy(family_mask[indices]).to(device)
            y = torch.from_numpy(target[indices].astype(np.float32, copy=False)).to(device)
            optimizer.zero_grad(set_to_none=True)
            if market_encoder == "tft":
                sequence = torch.from_numpy(market_sequence[indices]).to(device)
                padding = torch.from_numpy(sequence_padding_mask[indices]).to(device)
                logits = model(p, c, t, m, sequence, padding)
            else:
                logits = model(p, c, t, m)
            loss = F.binary_cross_entropy_with_logits(logits, y)
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
    model: nn.Module,
    price: np.ndarray,
    covariates: np.ndarray,
    family_text: np.ndarray,
    family_mask: np.ndarray,
    device: str,
    batch_size: int = 1024,
    market_encoder: str = "mlp",
    market_sequence: np.ndarray | None = None,
    sequence_padding_mask: np.ndarray | None = None,
) -> np.ndarray:
    _require_finite("fusion-inference price latents", price)
    _require_finite("fusion-inference covariates", covariates)
    _require_finite("fusion-inference text latents", family_text)
    _require_finite("fusion-inference family mask", family_mask)
    if family_text.ndim != 4 or family_mask.shape != family_text.shape[:3]:
        raise ValueError(
            "Article tokens/mask must have shapes (N, K, A, D) and (N, K, A)"
        )
    if market_encoder == "tft":
        if market_sequence is None or sequence_padding_mask is None:
            raise ValueError("TFT inference requires temporal sequences and masks")
        _require_finite("TFT inference sequences", market_sequence)
    model = model.to(device).eval()
    pieces = []
    for start in range(0, len(price), batch_size):
        stop = start + batch_size
        inputs = (
            torch.from_numpy(price[start:stop]).to(device),
            torch.from_numpy(covariates[start:stop]).to(device),
            torch.from_numpy(family_text[start:stop]).to(
                device=device, dtype=torch.float32
            ),
            torch.from_numpy(family_mask[start:stop]).to(device),
        )
        if market_encoder == "tft":
            logits = model(
                *inputs,
                torch.from_numpy(market_sequence[start:stop]).to(device),
                torch.from_numpy(sequence_padding_mask[start:stop]).to(device),
            )
        else:
            logits = model(*inputs)
        pieces.append(torch.sigmoid(logits).cpu().numpy())
    model.cpu()
    result = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
    _require_finite("fusion probabilities", result)
    return result


def _raw_text_batch(
    text_indices: dict[str, np.ndarray],
    stores: dict[str, RawEmbeddingStore],
    indices: np.ndarray | slice,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    articles, masks = {}, {}
    for family, token_index in text_indices.items():
        selected = token_index[indices]
        values = stores[family].gather(selected)
        articles[family] = torch.from_numpy(values).to(
            device=device, dtype=torch.float32
        )
        masks[family] = torch.from_numpy(selected >= 0).to(device)
    return articles, masks


def _date_group_ids(index: pl.DataFrame) -> np.ndarray:
    """Return one integer group per date while preserving row order."""
    if "date" not in index.columns:
        raise ValueError("Cross-stock attention requires a date column")
    if index["date"].null_count():
        raise ValueError("Cross-stock attention dates cannot be null")
    return np.ascontiguousarray(
        index["date"].cast(pl.Int32).to_numpy().astype(np.int64, copy=False)
    )


def _fusion_batches(
    row_count: int,
    batch_size: int,
    rng: np.random.Generator | None = None,
    group_ids: np.ndarray | None = None,
):
    """Yield row batches, keeping every date group intact when requested."""
    if group_ids is None:
        order = (
            rng.permutation(row_count)
            if rng is not None else np.arange(row_count)
        )
        for start in range(0, row_count, batch_size):
            yield order[start:start + batch_size]
        return
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1 or len(group_ids) != row_count:
        raise ValueError("group_ids must have one value per model row")
    unique_groups = np.unique(group_ids)
    if rng is not None:
        unique_groups = rng.permutation(unique_groups)
    pending: list[np.ndarray] = []
    pending_rows = 0
    for group in unique_groups:
        rows = np.flatnonzero(group_ids == group)
        if pending and pending_rows + len(rows) > batch_size:
            yield np.concatenate(pending)
            pending = []
            pending_rows = 0
        pending.append(rows)
        pending_rows += len(rows)
    if pending:
        yield np.concatenate(pending)


@torch.inference_mode()
def _raw_fusion_validation_metrics(
    model: nn.Module,
    price: np.ndarray,
    covariates: np.ndarray,
    text_indices: dict[str, np.ndarray],
    stores: dict[str, RawEmbeddingStore],
    target: np.ndarray,
    device: str,
    batch_size: int,
    market_encoder: str,
    market_sequence: np.ndarray | None,
    sequence_padding_mask: np.ndarray | None,
    stock_group_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate one epoch without moving the training model off its device."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    score = np.empty(len(target), dtype=np.float32)
    for batch_indices in _fusion_batches(
        len(target),
        batch_size,
        group_ids=stock_group_ids,
    ):
        articles, masks = _raw_text_batch(
            text_indices, stores, batch_indices, device
        )
        p = torch.from_numpy(price[batch_indices]).to(device)
        c = torch.from_numpy(covariates[batch_indices]).to(device)
        y = torch.from_numpy(
            target[batch_indices].astype(np.float32, copy=False)
        ).to(device)
        if market_encoder == "tft":
            group_tensor = (
                torch.from_numpy(stock_group_ids[batch_indices]).to(device)
                if stock_group_ids is not None else None
            )
            logits = model(
                p,
                c,
                articles,
                masks,
                torch.from_numpy(market_sequence[batch_indices]).to(device),
                torch.from_numpy(
                    sequence_padding_mask[batch_indices]
                ).to(device),
                group_tensor,
            )
        else:
            logits = model(p, c, articles, masks)
        per_example_loss = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        loss_sum += float(per_example_loss.sum().cpu())
        score[batch_indices] = torch.sigmoid(logits).cpu().numpy()
    if was_training:
        model.train()
    truth = target.astype(np.int8, copy=False)
    prediction = (score >= 0.5).astype(np.int8)
    optimized_threshold = _best_validation_threshold(truth, score)
    optimized_prediction = (
        score >= optimized_threshold
    ).astype(np.int8)
    return {
        "validation_bce": loss_sum / len(target),
        "validation_accuracy": float(accuracy_score(truth, prediction)),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "validation_roc_auc": float(roc_auc_score(truth, score)),
        "validation_optimized_balanced_accuracy": float(
            balanced_accuracy_score(truth, optimized_prediction)
        ),
        "validation_optimized_threshold": float(optimized_threshold),
        "validation_mean_probability": float(score.mean()),
        "validation_positive_rate": float(truth.mean()),
    }


def _plot_fold_training_diagnostics(
    history: pl.DataFrame,
    output_path: Path,
) -> None:
    """Plot selected-model learning curves for all walk-forward folds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is unavailable; saved training-history CSV but "
            "skipped the learning-curve plot"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.get_cmap("tab10")
    fold_numbers = history["fold"].unique().sort().to_list()
    for color_index, fold in enumerate(fold_numbers):
        part = history.filter(pl.col("fold") == fold).sort("epoch")
        epoch = part["epoch"].to_numpy()
        color = colors(color_index % 10)
        axes[0].plot(
            epoch, part["train_bce"].to_numpy(),
            color=color, linestyle="-", label=f"Fold {fold} train",
        )
        axes[0].plot(
            epoch, part["validation_bce"].to_numpy(),
            color=color, linestyle="--", label=f"Fold {fold} validation",
        )
        axes[1].plot(
            epoch, part["validation_accuracy"].to_numpy(),
            color=color, linestyle=":", label=f"Fold {fold} accuracy",
        )
        axes[1].plot(
            epoch, part["validation_balanced_accuracy"].to_numpy(),
            color=color, linestyle="-",
            label=f"Fold {fold} balanced accuracy",
        )
        axes[1].plot(
            epoch,
            part["validation_optimized_balanced_accuracy"].to_numpy(),
            color=color,
            linestyle="--",
            label=f"Fold {fold} optimized balanced accuracy",
        )
    axes[0].set_title("Training and validation BCE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[1].set_title("Validation directional metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_inner_selection_diagnostics(
    history: pl.DataFrame,
    output_path: Path,
) -> None:
    """Plot the selected trial's curves on the inner-validation split."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    required = {
        "epoch",
        "train_bce",
        "validation_bce",
        "validation_accuracy",
        "validation_balanced_accuracy",
    }
    if not required <= set(history.columns):
        raise ValueError(
            "Inner-selection history is missing diagnostic columns"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    part = history.sort("epoch")
    epoch = part["epoch"].to_numpy()
    axes[0].plot(
        epoch,
        part["train_bce"].to_numpy(),
        linestyle="-",
        label="Inner train",
    )
    axes[0].plot(
        epoch,
        part["validation_bce"].to_numpy(),
        linestyle="--",
        label="Inner validation",
    )
    axes[1].plot(
        epoch,
        part["validation_accuracy"].to_numpy(),
        linestyle="-",
        label="Validation accuracy",
    )
    axes[1].plot(
        epoch,
        part["validation_balanced_accuracy"].to_numpy(),
        linestyle="--",
        label="Validation balanced accuracy",
    )
    axes[0].set_title("Selected trial: BCE")
    axes[1].set_title("Selected trial: directional metrics")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_raw_fusion_model(
    price: np.ndarray,
    covariates: np.ndarray,
    text_indices: dict[str, np.ndarray],
    stores: dict[str, RawEmbeddingStore],
    target: np.ndarray,
    device: str,
    text_dim: int = 384,
    hidden_dim: int = 256,
    market_depth: int = 2,
    fusion_depth: int = 2,
    expansion: int = 2,
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    adapter_learning_rate_multiplier: float = 0.1,
    seed: int = 42,
    market_encoder: str = "mlp",
    market_sequence: np.ndarray | None = None,
    sequence_padding_mask: np.ndarray | None = None,
    market_attention_heads: int = 4,
    text_attention_heads: int = 4,
    text_attention_layers: int = 1,
    cross_stock_attention: bool = False,
    cross_stock_attention_heads: int = 4,
    stock_group_ids: np.ndarray | None = None,
    validation_price: np.ndarray | None = None,
    validation_covariates: np.ndarray | None = None,
    validation_text_indices: dict[str, np.ndarray] | None = None,
    validation_target: np.ndarray | None = None,
    validation_market_sequence: np.ndarray | None = None,
    validation_sequence_padding_mask: np.ndarray | None = None,
    validation_stock_group_ids: np.ndarray | None = None,
    select_best_checkpoint: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-5,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train raw-family adapters and all attention/fusion layers end to end."""
    _require_finite("raw-fusion price latents", price)
    _require_finite("raw-fusion covariates", covariates)
    _require_finite("raw-fusion targets", target)
    if market_encoder not in {"mlp", "tft"}:
        raise ValueError("market_encoder must be 'mlp' or 'tft'")
    if not 0.0 < adapter_learning_rate_multiplier <= 1.0:
        raise ValueError(
            "adapter_learning_rate_multiplier must be in (0, 1]"
        )
    if cross_stock_attention and market_encoder != "tft":
        raise ValueError("Cross-stock attention currently requires TFT")
    if cross_stock_attention and stock_group_ids is None:
        raise ValueError("Cross-stock training requires stock_group_ids")
    if not text_indices:
        raise ValueError("At least one raw embedding family is required")
    field_counts = {values.shape[1] for values in text_indices.values()}
    if len(field_counts) != 1:
        raise ValueError("Raw text families have inconsistent field counts")
    for family, values in text_indices.items():
        if values.ndim != 2 or len(values) != len(target):
            raise ValueError(f"Invalid raw text index shape for {family}")
        if family not in stores:
            raise ValueError(f"Missing raw embedding store for {family}")
    if market_encoder == "tft":
        if market_sequence is None or sequence_padding_mask is None:
            raise ValueError("TFT training requires temporal sequences and masks")
        _require_finite("TFT training sequences", market_sequence)
        if stock_group_ids is not None and len(stock_group_ids) != len(target):
            raise ValueError("Training stock_group_ids have the wrong length")
    validation_values = (
        validation_price,
        validation_covariates,
        validation_text_indices,
        validation_target,
    )
    has_validation = any(value is not None for value in validation_values)
    if has_validation and not all(value is not None for value in validation_values):
        raise ValueError("Validation arrays must be supplied together")
    if has_validation:
        _require_finite("raw-fusion validation price", validation_price)
        _require_finite(
            "raw-fusion validation covariates", validation_covariates
        )
        _require_finite("raw-fusion validation targets", validation_target)
        if set(validation_text_indices) != set(text_indices):
            raise ValueError(
                "Training and validation text families must be identical"
            )
        if market_encoder == "tft":
            if (
                validation_market_sequence is None
                or validation_sequence_padding_mask is None
            ):
                raise ValueError(
                    "TFT validation requires temporal sequences and masks"
                )
            _require_finite(
                "TFT validation sequences", validation_market_sequence
            )
            if cross_stock_attention and validation_stock_group_ids is None:
                raise ValueError(
                    "Cross-stock validation requires stock_group_ids"
                )
            if (
                validation_stock_group_ids is not None
                and len(validation_stock_group_ids) != len(validation_target)
            ):
                raise ValueError(
                    "Validation stock_group_ids have the wrong length"
                )
    family_dims = {
        family: stores[family].input_dim for family in text_indices
    }
    torch.manual_seed(seed)
    if market_encoder == "tft":
        model = RawTextTFTFusion(
            price.shape[1],
            covariates.shape[1],
            market_sequence.shape[2],
            family_dims,
            field_counts.pop(),
            text_dim,
            hidden_dim,
            fusion_depth=fusion_depth,
            expansion=expansion,
            market_attention_heads=market_attention_heads,
            text_attention_heads=text_attention_heads,
            text_attention_layers=text_attention_layers,
            cross_stock_attention=cross_stock_attention,
            cross_stock_attention_heads=cross_stock_attention_heads,
            dropout=dropout,
        ).to(device)
    else:
        model = RawTextResidualFusion(
            price.shape[1],
            covariates.shape[1],
            family_dims,
            field_counts.pop(),
            text_dim,
            hidden_dim,
            market_depth=market_depth,
            fusion_depth=fusion_depth,
            expansion=expansion,
            text_attention_heads=text_attention_heads,
            text_attention_layers=text_attention_layers,
            dropout=dropout,
        ).to(device)
    adapter_parameters = list(
        model.text_attention.adapters.parameters()
    )
    adapter_parameter_ids = {
        id(parameter) for parameter in adapter_parameters
    }
    remaining_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in adapter_parameter_ids
    ]
    optimizer = torch.optim.AdamW([
        {
            "params": remaining_parameters,
            "lr": learning_rate,
        },
        {
            "params": adapter_parameters,
            "lr": learning_rate * adapter_learning_rate_multiplier,
        },
    ], weight_decay=weight_decay)
    positive_count = int((target > 0.5).sum())
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_validation_metric = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_bce_sum = 0.0
        train_example_count = 0
        for batch_indices in _fusion_batches(
            len(target),
            batch_size,
            rng=rng,
            group_ids=stock_group_ids if cross_stock_attention else None,
        ):
            p = torch.from_numpy(price[batch_indices]).to(device)
            c = torch.from_numpy(covariates[batch_indices]).to(device)
            y = torch.from_numpy(
                target[batch_indices].astype(np.float32, copy=False)
            ).to(device)
            articles, masks = _raw_text_batch(
                text_indices, stores, batch_indices, device
            )
            optimizer.zero_grad(set_to_none=True)
            if market_encoder == "tft":
                temporal = torch.from_numpy(
                    market_sequence[batch_indices]
                ).to(device)
                padding = torch.from_numpy(
                    sequence_padding_mask[batch_indices]
                ).to(device)
                group_tensor = (
                    torch.from_numpy(stock_group_ids[batch_indices]).to(device)
                    if cross_stock_attention else None
                )
                logits = model(
                    p,
                    c,
                    articles,
                    masks,
                    temporal,
                    padding,
                    group_tensor,
                )
            else:
                logits = model(p, c, articles, masks)
            per_example_loss = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = per_example_loss.mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Raw fusion loss became non-finite in epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_bce_sum += float(
                per_example_loss.detach().sum().cpu()
            )
            train_example_count += len(batch_indices)
        train_bce = train_bce_sum / max(train_example_count, 1)
        epoch_metrics = {
            "epoch": float(epoch),
            "bce": float(train_bce),
            "train_bce": float(train_bce),
            "train_positive_rate": float(positive_count / len(target)),
        }
        if has_validation:
            epoch_metrics.update(_raw_fusion_validation_metrics(
                model,
                validation_price,
                validation_covariates,
                validation_text_indices,
                stores,
                validation_target,
                device,
                batch_size,
                market_encoder,
                validation_market_sequence,
                validation_sequence_padding_mask,
                (
                    validation_stock_group_ids
                    if cross_stock_attention else None
                ),
            ))
        history.append(epoch_metrics)
        if has_validation and select_best_checkpoint:
            current_validation_metric = epoch_metrics["validation_bce"]
            if (
                current_validation_metric
                < best_validation_metric - float(early_stopping_min_delta)
            ):
                best_validation_metric = current_validation_metric
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= int(early_stopping_patience):
                break
    model = model.cpu()
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch is None:
        best_epoch = len(history)
    for row in history:
        row["best_epoch"] = float(best_epoch)
        row["is_best_epoch"] = float(row["epoch"] == best_epoch)
        row["stopped_early"] = float(len(history) < epochs)
    return model.eval(), history


@torch.inference_mode()
def predict_raw_fusion(
    model: nn.Module,
    price: np.ndarray,
    covariates: np.ndarray,
    text_indices: dict[str, np.ndarray],
    stores: dict[str, RawEmbeddingStore],
    device: str,
    batch_size: int = 256,
    market_encoder: str = "mlp",
    market_sequence: np.ndarray | None = None,
    sequence_padding_mask: np.ndarray | None = None,
    cross_stock_attention: bool = False,
    stock_group_ids: np.ndarray | None = None,
) -> np.ndarray:
    _require_finite("raw-fusion inference price", price)
    _require_finite("raw-fusion inference covariates", covariates)
    if cross_stock_attention and market_encoder != "tft":
        raise ValueError("Cross-stock attention currently requires TFT")
    if cross_stock_attention and stock_group_ids is None:
        raise ValueError("Cross-stock inference requires stock_group_ids")
    model = model.to(device).eval()
    result = np.empty(len(price), dtype=np.float32)
    for batch_indices in _fusion_batches(
        len(price),
        batch_size,
        group_ids=stock_group_ids if cross_stock_attention else None,
    ):
        articles, masks = _raw_text_batch(
            text_indices, stores, batch_indices, device
        )
        p = torch.from_numpy(price[batch_indices]).to(device)
        c = torch.from_numpy(covariates[batch_indices]).to(device)
        if market_encoder == "tft":
            logits = model(
                p,
                c,
                articles,
                masks,
                torch.from_numpy(market_sequence[batch_indices]).to(device),
                torch.from_numpy(
                    sequence_padding_mask[batch_indices]
                ).to(device),
                (
                    torch.from_numpy(stock_group_ids[batch_indices]).to(device)
                    if cross_stock_attention else None
                ),
            )
        else:
            logits = model(p, c, articles, masks)
        result[batch_indices] = torch.sigmoid(logits).cpu().numpy()
    model.cpu()
    _require_finite("raw-fusion probabilities", result)
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
    # Import TimesFM's reversible normalization and online patch-statistics
    # utilities. They operate on torch tensors without training any parameters.
    from timesfm.torch.util import revin, update_running_stats

    module = wrapper.model  # Frozen TimesFM torch module.
    patch_length = module.p  # P: number of scalar observations per TimesFM patch.

    # Truncate each row to its most recent lookback values and store it as a
    # one-dimensional float32 array with shape (T_i,), where T_i <= lookback.
    series_by_row = [
        np.asarray(record["target"][-lookback:], dtype=np.float32)
        for record in records
    ]

    # Map each required patch count M_i=ceil(T_i/P) to the original row
    # positions having that count. Equal-length patch grids can share a batch.
    groups: dict[int, list[int]] = {}
    for row, series in enumerate(series_by_row):
        # An empty series cannot produce even one partially observed patch.
        if not len(series):
            raise ValueError("Cannot extract a TimesFM latent from an empty context")
        patch_count = int(np.ceil(len(series) / patch_length))  # Scalar M_i.
        groups.setdefault(patch_count, []).append(row)  # Append original row index.

    # A shared LOOKBACK-width tensor creates fully padded prefix patches for
    # short histories. Those tokens can contaminate later valid transformer
    # states with NaNs. Grouping by actual patch count retains batching while
    # ensuring that every patch contains at least one observation.
    pooled_by_row: list[np.ndarray | None] = [None] * len(records)  # N output slots.
    for patch_count, row_indices in groups.items():
        # B_g: rows in this group; M: patch_count; P: patch_length.
        # The padded scalar width is exactly M*P, so only the first patch can
        # be partially padded and no patch is fully padded.
        context_width = patch_count * patch_length  # Scalar M*P.

        # Right-aligned scalar contexts. `True` in masks denotes padding.
        values = np.zeros((len(row_indices), context_width), dtype=np.float32)
        masks = np.ones((len(row_indices), context_width), dtype=bool)
        for group_row, original_row in enumerate(row_indices):
            series = series_by_row[original_row]  # (T_i,).
            values[group_row, -len(series):] = series  # Fill observed suffix.
            masks[group_row, -len(series):] = False  # Mark observed suffix valid.

        inputs = torch.from_numpy(values).to(device)  # (B_g, M*P), float32.
        input_masks = torch.from_numpy(masks).to(device)  # (B_g, M*P), bool.
        valid = (~input_masks).to(inputs.dtype)  # (B_g, M*P), 1=observed.
        count = valid.sum(1, keepdim=True).clamp_min(1.0)  # (B_g, 1).
        mean = (inputs * valid).sum(1, keepdim=True) / count  # (B_g, 1).
        variance = (((inputs - mean) ** 2) * valid).sum(1, keepdim=True) / count
        # variance and its square root have shape (B_g, 1).
        inputs = revin(
            inputs, mean, variance.sqrt(), reverse=False
        )  # Globally normalized scalar context: (B_g, M*P).
        inputs = torch.where(
            input_masks, 0.0, inputs
        )  # Keep padded scalar positions exactly zero: (B_g, M*P).

        # Split the scalar axis into TimesFM's patch grid.
        patched_inputs = inputs.reshape(len(row_indices), patch_count, patch_length)
        # patched_inputs: (B_g, M, P).
        patched_masks = input_masks.reshape(len(row_indices), patch_count, patch_length)
        # patched_masks: (B_g, M, P), True=padding.

        # Initialize per-row online statistics accumulated from patch 0 through
        # the current patch. Each state tensor has shape (B_g,).
        n = torch.zeros(len(row_indices), device=inputs.device)  # Valid count.
        mu = torch.zeros_like(n)  # Running mean.
        sigma = torch.zeros_like(n)  # Running scale/standard deviation.
        patch_mu, patch_sigma = [], []  # M snapshots, each shaped (B_g,).
        for patch in range(patch_count):
            # Update each row's running statistics using patch `patch`, while
            # ignoring scalar positions whose patch mask is True.
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, patch], patched_masks[:, patch]
            )
            patch_mu.append(mu)  # Running mean after this patch: (B_g,).
            patch_sigma.append(sigma)  # Running scale after this patch: (B_g,).

        # Stack the per-patch running statistics along the patch axis and apply
        # TimesFM's causal patch-wise normalization.
        normalized = revin(
            patched_inputs,  # (B_g, M, P).
            torch.stack(patch_mu, 1),  # (B_g, M).
            torch.stack(patch_sigma, 1),  # (B_g, M).
            reverse=False,
        )  # Normalized patches: (B_g, M, P).
        normalized = torch.where(
            patched_masks, 0.0, normalized
        )  # Zero padded scalar positions again: (B_g, M, P).

        # Run the frozen TimesFM backbone. Only hidden_tokens is retained;
        # D_price is the model's hidden width.
        (_, hidden_tokens, _, _), _ = module(
            normalized, patched_masks
        )  # hidden_tokens: (B_g, M, D_price).

        # A token is valid when its patch contains at least one observed scalar.
        valid_tokens = (~patched_masks).any(-1)  # (B_g, M), bool.

        # Replace invalid token vectors with zeros before summation, preventing
        # padded hidden states from contributing to the pooled representation.
        masked_hidden = torch.where(
            valid_tokens.unsqueeze(-1), hidden_tokens, torch.zeros_like(hidden_tokens)
        )  # (B_g, M, D_price).

        # Fail immediately if TimesFM produced NaN or infinity in any retained
        # or zero-masked hidden position.
        if not torch.isfinite(masked_hidden).all():
            raise ValueError(
                "TimesFM produced a non-finite hidden state after removing "
                "fully padded patches"
            )

        pooled = masked_hidden.sum(1)  # Sum valid tokens: (B_g, D_price).
        pooled = pooled / valid_tokens.sum(
            1, keepdim=True
        ).clamp_min(1).to(pooled.dtype)  # Mean valid tokens: (B_g, D_price).
        pooled_numpy = pooled.float().cpu().numpy()  # Host float32 (B_g, D_price).

        # Restore each group result to its position in the original records.
        for group_row, original_row in enumerate(row_indices):
            pooled_by_row[original_row] = pooled_numpy[group_row]  # (D_price,).

    # Every input record must have received exactly one pooled group result.
    if any(value is None for value in pooled_by_row):
        raise RuntimeError("TimesFM pooling did not return every requested row")

    # Stack row vectors into the cache-ready latent matrix (N, D_price).
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
                    "article_aggregation": (
                        "preserve_text_fields_then_fold_local_market_attention"
                    ),
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
        # Versioned name prevents reuse of the former irreversible row mean.
        row_path = (
            scope_dir / "row_text_latents" /
            f"{family}_field_article_tokens_v2.parquet"
        )
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
    text_fields: Sequence[str],
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
        One parquet path per embedding family. Each table contains one token
        per stock-date/text-field pair named ``<family>_latent_*``.
    families:
        Ordered families included in this model variant. Their order becomes
        the family axis of the returned text tensor.
    text_fields:
        Stable ordered text-field catalog. Its order becomes the article/field
        axis of the returned text tensor.
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
        Contiguous float16 storage tensor with shape ``(N, K, A, D_text)``,
        where ``K`` is the embedding-family count and ``A`` is the number of
        distinct text fields. Mini-batches are promoted to float32 on device.
    mask:
        Float32 availability mask with shape ``(N, K, A)``. A value of one
        means the family supplied that text field for the row.
    target:
        Float32 direction-label vector with shape ``(N,)``. It may contain
        NaN values only when ``require_target=False``.

    Notes
    -----
    The join with ``price_latents`` is inner because a fusion example cannot
    be constructed without a price representation. Missing text fields retain
    the row and become zero tokens with a zero availability mask.
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
    if require_target:
        frame = frame.filter(pl.col("target_up").is_not_null())
    # Stable chronological ordering makes saved predictions easy to audit.
    frame = frame.sort(["date", "ticker"])

    if not text_fields:
        raise ValueError("text_fields cannot be empty")
    family_tables: list[tuple[pl.DataFrame, list[str]]] = []
    text_dim: int | None = None
    for family in families:
        family_frame = pl.read_parquet(row_text_paths[family]).with_columns(
            pl.col("row_id").cast(pl.UInt64)
        )
        if "text_field" not in family_frame.columns:
            raise ValueError(
                f"{row_text_paths[family]} is an obsolete mean-pooled cache; "
                "regenerate field article tokens"
            )
        family_columns = sorted(
            c for c in family_frame.columns if c.startswith(f"{family}_latent_")
        )
        if not family_columns:
            raise ValueError(f"No latent columns found for {family}")
        if text_dim is None:
            text_dim = len(family_columns)
        elif len(family_columns) != text_dim:
            raise ValueError("Text families have inconsistent latent dimensions")
        family_tables.append((family_frame, family_columns))

    # Float16 halves host memory for the at-most-21 field tokens per row. Each
    # training/inference mini-batch is explicitly promoted to float32.
    text = np.zeros(
        (frame.height, len(families), len(text_fields), int(text_dim)),
        dtype=np.float16,
    )
    mask = np.zeros(
        (frame.height, len(families), len(text_fields)), dtype=np.float32
    )
    row_lookup = frame.select("row_id").with_row_index("_row_index")
    field_lookup = pl.DataFrame({
        "text_field": list(text_fields),
        "_field_index": np.arange(len(text_fields), dtype=np.int32),
    })
    for family_index, (family_frame, family_columns) in enumerate(family_tables):
        aligned = (
            family_frame.select(["row_id", "text_field", *family_columns])
            .join(row_lookup, on="row_id", how="inner", validate="m:1")
            .join(field_lookup, on="text_field", how="inner", validate="m:1")
        )
        if not aligned.height:
            continue
        row_index = aligned["_row_index"].to_numpy()
        field_index = aligned["_field_index"].to_numpy()
        values = aligned.select(family_columns).to_numpy().astype(
            np.float16, copy=False
        )
        text[row_index, family_index, field_index] = values
        mask[row_index, family_index, field_index] = 1.0

    # Convert the remaining modalities to the layouts expected by the models.
    price = np.ascontiguousarray(frame.select(price_columns).to_numpy().astype(np.float32))
    text = np.ascontiguousarray(text)
    mask = np.ascontiguousarray(mask)
    target = frame["target_up"].to_numpy().astype(np.float32)
    _require_finite("TimesFM price latents", price)
    _require_finite("text-coder latents", text)
    _require_finite("text-family availability mask", mask)
    if require_target:
        _require_finite("training targets", target)
    return frame.select(["row_id", "date", "ticker", "target_up"]), price, text, mask, target


def _assemble_raw_fusion_arrays(
    row_ids: pl.DataFrame,
    price_latents: pl.DataFrame,
    targets: pl.DataFrame,
    links: pl.DataFrame,
    stores: dict[str, RawEmbeddingStore],
    families: Sequence[str],
    text_fields: Sequence[str],
    require_target: bool = True,
) -> tuple[
    pl.DataFrame,
    np.ndarray,
    dict[str, np.ndarray],
    None,
    np.ndarray,
]:
    """Align rows to raw embedding-matrix indices without loading raw vectors."""
    scoped_ids = row_ids.select(pl.col("row_id").cast(pl.UInt64)).unique()
    scoped_price = price_latents.with_columns(pl.col("row_id").cast(pl.UInt64))
    scoped_targets = targets.select([
        pl.col("row_id").cast(pl.UInt64), "target_up",
    ])
    frame = (
        scoped_ids.join(scoped_price, on="row_id", how="inner")
        .join(scoped_targets, on="row_id", how="left")
    )
    if require_target:
        frame = frame.filter(pl.col("target_up").is_not_null())
    frame = frame.sort(["date", "ticker"])
    price_columns = sorted(
        column for column in frame.columns if column.startswith("price_latent_")
    )
    row_lookup = frame.select("row_id").with_row_index("_row_index")
    field_lookup = pl.DataFrame({
        "text_field": list(text_fields),
        "_field_index": np.arange(len(text_fields), dtype=np.int32),
    })
    scoped_links = (
        links.select([
            pl.col("row_id").cast(pl.UInt64), "text_field", "text_id",
        ])
        .unique()
        .join(row_lookup, on="row_id", how="inner", validate="m:1")
        .join(field_lookup, on="text_field", how="inner", validate="m:1")
    )
    text_indices: dict[str, np.ndarray] = {}
    for family in families:
        indices = np.full(
            (frame.height, len(text_fields)), -1, dtype=np.int64
        )
        aligned = scoped_links.join(
            stores[family].index,
            on="text_id",
            how="inner",
            validate="m:1",
        )
        if aligned.height:
            rows = aligned["_row_index"].to_numpy()
            fields = aligned["_field_index"].to_numpy()
            has_duplicate_slot = aligned.select(
                pl.struct(["_row_index", "_field_index"]).is_duplicated().any()
            ).item()
            if has_duplicate_slot:
                raise ValueError(
                    f"Multiple {family} articles occupy one row/text-field slot"
                )
            indices[rows, fields] = aligned["embedding_row"].to_numpy()
        text_indices[family] = np.ascontiguousarray(indices)
    price = np.ascontiguousarray(
        frame.select(price_columns).to_numpy().astype(np.float32)
    )
    target = frame["target_up"].to_numpy().astype(np.float32)
    _require_finite("TimesFM price latents", price)
    if require_target:
        _require_finite("training targets", target)
    return (
        frame.select(["row_id", "date", "ticker", "target_up"]),
        price,
        text_indices,
        None,
        target,
    )


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
            pl.col("decision_threshold").mean().alias("decision_threshold"),
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


def _temporal_covariate_matrix(
    index: pl.DataFrame,
    features: pl.DataFrame,
    columns: Sequence[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build right-aligned per-ticker histories ending at each forecast row."""
    if lookback < 2:
        raise ValueError("TFT lookback must be at least two time steps")
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(f"Missing TFT temporal covariates: {missing}")
    histories: dict[str, np.ndarray] = {}
    positions: dict[int, tuple[str, int]] = {}
    selected = ["row_id", "ticker", "date", *columns]
    for key, frame in features.select(selected).sort(
        ["ticker", "date"]
    ).partition_by("ticker", as_dict=True).items():
        ticker = key[0] if isinstance(key, tuple) else key
        values = frame.select([
            pl.col(column).cast(pl.Float64).fill_null(float("nan"))
            for column in columns
        ]).to_numpy().astype(np.float32, copy=False)
        histories[ticker] = values
        for position, row_id in enumerate(frame["row_id"].to_list()):
            positions[int(row_id)] = (ticker, position)

    sequences = np.zeros(
        (index.height, lookback, len(columns)), dtype=np.float32
    )
    padding_mask = np.ones((index.height, lookback), dtype=np.bool_)
    for output_row, row_id in enumerate(index["row_id"].to_list()):
        location = positions.get(int(row_id))
        if location is None:
            raise ValueError(f"No prepared feature history for row_id={row_id}")
        ticker, position = location
        start = max(0, position + 1 - lookback)
        window = histories[ticker][start:position + 1]
        sequences[output_row, -len(window):] = window
        padding_mask[output_row, -len(window):] = False
    return np.ascontiguousarray(sequences), np.ascontiguousarray(padding_mask)


def _fit_temporal_scaler(
    values: np.ndarray, padding_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Fit temporal imputation/scaling on valid training timesteps only."""
    valid = values[~padding_mask]
    if not len(valid):
        raise ValueError("Cannot fit a TFT scaler without valid timesteps")
    return _fit_covariate_scaler(valid)


def _apply_temporal_scaler(
    values: np.ndarray,
    padding_mask: np.ndarray,
    scaler: dict[str, np.ndarray],
) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    scaled = _apply_covariate_scaler(flat, scaler).reshape(values.shape)
    scaled[padding_mask] = 0.0
    return np.ascontiguousarray(scaled)


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


def _best_validation_threshold(
    truth: np.ndarray,
    score: np.ndarray,
) -> float:
    """Find the exact balanced-accuracy threshold, preferring 0.5 on ties."""
    truth = np.asarray(truth, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if truth.ndim != 1 or score.ndim != 1 or len(truth) != len(score):
        raise ValueError("Threshold truth and score must be aligned vectors")
    if len(np.unique(truth)) != 2:
        raise ValueError("Threshold selection requires both target classes")
    if not np.isfinite(score).all():
        raise ValueError("Threshold scores must be finite")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        truth, score, drop_intermediate=False
    )
    finite = np.isfinite(thresholds)
    thresholds = thresholds.copy()
    thresholds[~finite] = np.nextafter(score.max(), np.inf)
    balanced_accuracy = 0.5 * (
        true_positive_rate + 1.0 - false_positive_rate
    )
    best = np.flatnonzero(np.isclose(
        balanced_accuracy,
        balanced_accuracy.max(),
        rtol=0.0,
        atol=1e-12,
    ))
    return float(thresholds[
        best[np.argmin(np.abs(thresholds[best] - 0.5))]
    ])


def _dataframe_fingerprint(
    frame: pl.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    """Hash row contents without making the digest depend on row order."""
    selected = frame.select(columns) if columns is not None else frame
    row_hashes = np.sort(selected.hash_rows(seed=0).to_numpy())
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def _purged_inner_positions(
    index: pl.DataFrame,
    purge_dates: int,
    validation_fraction: float = 0.20,
    minimum_train_dates: int = 60,
    minimum_validation_dates: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a chronological inner train/validation split with a date purge."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Inner validation fraction must be between zero and one")
    dates = index["date"].unique().sort().to_list()
    validation_dates = max(
        int(minimum_validation_dates),
        int(np.ceil(len(dates) * validation_fraction)),
    )
    maximum_validation_dates = (
        len(dates) - int(purge_dates) - int(minimum_train_dates)
    )
    validation_dates = min(validation_dates, maximum_validation_dates)
    if validation_dates < minimum_validation_dates:
        raise ValueError(
            "Not enough dates for a purged inner train/validation split"
        )
    validation_start_position = len(dates) - validation_dates
    train_end_position = validation_start_position - int(purge_dates)
    if train_end_position < minimum_train_dates:
        raise ValueError("Inner training period is too short after purging")
    train_end_date = dates[train_end_position - 1]
    validation_start_date = dates[validation_start_position]
    validation_end_date = dates[-1]
    row_dates = index["date"].to_numpy()
    train_positions = np.flatnonzero(row_dates <= train_end_date)
    validation_positions = np.flatnonzero(row_dates >= validation_start_date)
    if not len(train_positions) or not len(validation_positions):
        raise ValueError("Purged inner split produced an empty partition")
    manifest = {
        "inner_train_rows": int(len(train_positions)),
        "inner_validation_rows": int(len(validation_positions)),
        "inner_train_dates": int(train_end_position),
        "purged_dates": int(purge_dates),
        "inner_validation_dates": int(validation_dates),
        "inner_train_end": train_end_date,
        "inner_validation_start": validation_start_date,
        "inner_validation_end": validation_end_date,
    }
    return train_positions, validation_positions, manifest


def _slice_raw_fusion_arrays(
    arrays: tuple,
    positions: np.ndarray,
) -> tuple:
    """Select rows from assembled raw-fusion arrays without changing order."""
    index, price, text_indices, placeholder, target = arrays
    return (
        index[positions],
        np.ascontiguousarray(price[positions]),
        {
            family: np.ascontiguousarray(values[positions])
            for family, values in text_indices.items()
        },
        placeholder,
        np.ascontiguousarray(target[positions]),
    )


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
    fold_assignments: pl.DataFrame | None,
    requested_families: Sequence[str],
    covariate_columns: Sequence[str],
    device: str,
    fusion_epochs: int = 10,
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
    market_encoder: str = "mlp",
    temporal_covariate_columns: Sequence[str] = (),
    temporal_lookback: int = 32,
    temporal_lookback_candidates: Sequence[int] = (),
    tft_attention_heads: int = 4,
    raw_text_dim: int = 384,
    text_attention_heads: int = 4,
    text_attention_layers: int = 1,
    cross_stock_attention: bool = False,
    cross_stock_attention_heads: int = 4,
    run_outer_folds: bool = True,
) -> dict[str, pl.DataFrame]:
    """Optionally run nested outer folds, then select and refit on all data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_test_path = Path(raw_test_path or data_dir / "test.parquet")
    if fusion_epochs < 1:
        raise ValueError("fusion_epochs must be positive")
    if fusion_batch_size < 1:
        raise ValueError("fusion_batch_size must be positive")
    if fusion_hidden_dim < 1:
        raise ValueError("fusion_hidden_dim must be positive")
    if min(market_depth, fusion_depth) < 0:
        raise ValueError("Network depths cannot be negative")
    if residual_expansion < 1:
        raise ValueError("residual_expansion must be positive")
    if not 0.0 <= fusion_dropout < 1.0:
        raise ValueError("fusion_dropout must be in [0, 1)")
    if tuning_trials < 0:
        raise ValueError("tuning_trials cannot be negative")
    if forecast_horizon_weekdays < 1:
        raise ValueError("forecast_horizon_weekdays must be positive")
    if not submission_years:
        raise ValueError("submission_years cannot be empty")
    if expected_submission_rows < 1:
        raise ValueError("expected_submission_rows must be positive")
    if expected_submission_rows % len(submission_years):
        raise ValueError(
            "expected_submission_rows must be divisible by submission years"
        )
    if raw_text_dim < 1 or text_attention_heads < 1:
        raise ValueError("Raw-text dimensions and attention heads must be positive")
    if raw_text_dim % text_attention_heads:
        raise ValueError(
            "raw_text_dim must be divisible by text_attention_heads"
        )
    if text_attention_layers < 1:
        raise ValueError("text_attention_layers must be positive")
    if market_encoder not in {"mlp", "tft"}:
        raise ValueError("market_encoder must be 'mlp' or 'tft'")
    if cross_stock_attention and market_encoder != "tft":
        raise ValueError("Cross-stock attention currently requires TFT")
    if cross_stock_attention and cross_stock_attention_heads < 1:
        raise ValueError("cross_stock_attention_heads must be positive")
    if market_encoder == "tft" and not temporal_covariate_columns:
        raise ValueError("TFT requires temporal_covariate_columns")
    if market_encoder == "tft" and tft_attention_heads < 1:
        raise ValueError("tft_attention_heads must be positive")
    if market_encoder == "tft":
        lookback_candidates = tuple(sorted(set(
            int(value) for value in (
                temporal_lookback_candidates or (temporal_lookback,)
            )
        )))
        if not lookback_candidates or any(value < 2 for value in lookback_candidates):
            raise ValueError("Every TFT lookback candidate must be at least two")
        if int(temporal_lookback) not in lookback_candidates:
            raise ValueError(
                "The default TFT lookback must be included in its candidates"
            )
        max_temporal_lookback = max(lookback_candidates)
    else:
        lookback_candidates = ()
        max_temporal_lookback = int(temporal_lookback)
    hidden_dim_candidates = tuple(
        hidden_dim for hidden_dim in (64, 128, 256)
        if (
            market_encoder != "tft"
            or (
                hidden_dim % tft_attention_heads == 0
                and (
                    not cross_stock_attention
                    or hidden_dim % cross_stock_attention_heads == 0
                )
            )
        )
    )
    if tune_hyperparameters and tuning_trials and not hidden_dim_candidates:
        raise ValueError(
            "No tuned hidden dimension is divisible by tft_attention_heads"
        )
    if (
        market_encoder == "tft"
        and (not tune_hyperparameters or tuning_trials == 0)
        and (
            fusion_hidden_dim % tft_attention_heads
            or (
                cross_stock_attention
                and fusion_hidden_dim % cross_stock_attention_heads
            )
        )
    ):
        raise ValueError(
            "fusion_hidden_dim must be divisible by every TFT attention head count"
        )
    if cross_stock_attention:
        feature_set = (
            "timesfm_temporal_tft_cross_stock_unified_raw_text_attention"
        )
    elif market_encoder == "tft":
        feature_set = "timesfm_temporal_tft_unified_raw_text_attention"
    else:
        feature_set = "timesfm_covariates_unified_raw_text_attention"
    if run_outer_folds:
        if fold_assignments is None:
            raise ValueError(
                "fold_assignments are required when run_outer_folds=True"
            )
        required_fold_columns = {"fold", "split", "row_id", "date"}
        missing_fold_columns = sorted(
            required_fold_columns - set(fold_assignments.columns)
        )
        if missing_fold_columns:
            raise ValueError(
                f"Fold assignments are missing columns: {missing_fold_columns}"
            )
        if fold_assignments.select(
            pl.any_horizontal(
                pl.col(column).is_null()
                for column in required_fold_columns
            ).any()
        ).item():
            raise ValueError("Fold assignments contain null required values")
        if fold_assignments.select(
            pl.struct(["fold", "split", "row_id"]).is_duplicated().any()
        ).item():
            raise ValueError("Fold assignments contain duplicate rows")
        fold_numbers = fold_assignments["fold"].unique().sort().to_list()
        if not fold_numbers or any(fold is None for fold in fold_numbers):
            raise ValueError("No outer folds were found")
        for fold in fold_numbers:
            fold_rows = fold_assignments.filter(pl.col("fold") == fold)
            unknown_splits = sorted(
                set(fold_rows["split"].drop_nulls().to_list())
                - {"train", "validation"}
            )
            if unknown_splits:
                raise ValueError(
                    f"Fold {fold} has unsupported splits: {unknown_splits}"
                )
            train_rows = fold_rows.filter(pl.col("split") == "train")
            validation_rows = fold_rows.filter(
                pl.col("split") == "validation"
            )
            if train_rows.is_empty() or validation_rows.is_empty():
                raise ValueError(
                    f"Fold {fold} requires non-empty train and validation rows"
                )
            overlap = train_rows.select("row_id").join(
                validation_rows.select("row_id"),
                on="row_id",
                how="inner",
            )
            if not overlap.is_empty():
                raise ValueError(
                    f"Fold {fold} has rows in both train and validation"
                )
            if train_rows["date"].max() >= validation_rows["date"].min():
                raise ValueError(
                    f"Fold {fold} is not chronologically ordered"
                )
    else:
        fold_numbers = []
    (output_dir / "training_mode.json").write_text(json.dumps({
        "run_outer_folds": bool(run_outer_folds),
        "mode": "nested-folds" if run_outer_folds else "full-only",
        "final_selection": "purged_inner_validation",
        "inner_validation_splits": 1,
        "training_loss": "binary_cross_entropy",
        "checkpoint_metric": "validation_bce",
        "optuna_objective": "negative_validation_bce",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-5,
        "epoch_selection": (
            "post_optuna_inner_validation"
            if tune_hyperparameters and tuning_trials > 0 else
            "fixed_configuration_inner_validation"
        ),
        "threshold_selection": "exact_inner_validation_balanced_accuracy",
        "cross_stock_attention": bool(cross_stock_attention),
    }, indent=2))
    if not covariate_columns:
        raise ValueError("covariate_columns cannot be empty")
    if "text_field" not in train_links.columns or "text_field" not in test_links.columns:
        raise ValueError("Text links must include text_field for article attention")
    text_fields = sorted(set(
        train_links["text_field"].drop_nulls().to_list()
    ) | set(
        test_links["text_field"].drop_nulls().to_list()
    ))
    if not text_fields:
        raise ValueError("No text fields are available for article attention")
    pl.DataFrame({
        "field_index": np.arange(len(text_fields), dtype=np.int16),
        "text_field": text_fields,
    }).write_csv(output_dir / "text_field_catalog.csv")

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
    raw_cache_dir = output_dir.parent / "raw_embedding_memmaps"
    stores = {
        family: prepare_raw_embedding_store(
            family=family,
            embedding_path=data_dir / f"{family}_textemb.parquet",
            cache_dir=raw_cache_dir,
        )
        for family in families
    }
    variants = {family: (family,) for family in families}
    if len(families) > 1:
        variants["all_families"] = tuple(families)
    tuning_variant = "all_families" if "all_families" in variants else families[0]
    fingerprint_feature_columns = list(dict.fromkeys([
        "row_id",
        "date",
        "ticker",
        *covariate_columns,
        *temporal_covariate_columns,
    ]))
    training_data_fingerprints = {
        "features": _dataframe_fingerprint(
            train_features, fingerprint_feature_columns
        ),
        "price_latents": _dataframe_fingerprint(train_price_latents),
        "targets": _dataframe_fingerprint(
            train_targets,
            [
                column for column in (
                    "row_id",
                    "date",
                    "ticker",
                    "target_up",
                )
                if column in train_targets.columns
            ],
        ),
        "text_links": _dataframe_fingerprint(
            train_links,
            [
                column for column in (
                    "row_id",
                    "text_field",
                    "text_id",
                )
                if column in train_links.columns
            ],
        ),
        "fold_assignments": (
            _dataframe_fingerprint(fold_assignments)
            if fold_assignments is not None else None
        ),
    }
    study_signature = hashlib.sha256(json.dumps({
        "selection_protocol_version": 7,
        "market_encoder": market_encoder,
        "families": families,
        "family_dims": {
            family: stores[family].input_dim for family in families
        },
        "embedding_sources": {
            family: {
                "size": (data_dir / f"{family}_textemb.parquet").stat().st_size,
                "mtime_ns": (
                    data_dir / f"{family}_textemb.parquet"
                ).stat().st_mtime_ns,
            }
            for family in families
        },
        "covariates": list(covariate_columns),
        "temporal_covariates": list(temporal_covariate_columns),
        "temporal_lookback_candidates": list(lookback_candidates),
        "forecast_horizon_weekdays": int(forecast_horizon_weekdays),
        "fusion_epochs": int(fusion_epochs),
        "fusion_batch_size": int(fusion_batch_size),
        "raw_text_dim": int(raw_text_dim),
        "tft_attention_heads": int(tft_attention_heads),
        "text_attention_heads": int(text_attention_heads),
        "text_attention_layers": int(text_attention_layers),
        "cross_stock_attention": bool(cross_stock_attention),
        "cross_stock_attention_heads": int(
            cross_stock_attention_heads
        ),
        "seed": int(seed),
        "optuna_pruning": "none",
        "inner_validation_splits": 1,
        "checkpoint_metric": "validation_bce",
        "optuna_objective": "negative_validation_bce",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-5,
        "training_loss": "binary_cross_entropy",
        "adapter_learning_rate_multiplier": 0.1,
        "epoch_selection": "post_optuna_inner_validation",
        "threshold_selection": "exact_inner_validation_balanced_accuracy",
        "training_data_fingerprints": training_data_fingerprints,
        "train_target_rows": int(train_targets.height),
        "train_price_latent_rows": int(train_price_latents.height),
    }, sort_keys=True).encode()).hexdigest()[:12]

    # Raw embedding adapters, text self-attention, cross-attention, and fusion
    # are all trained jointly inside each fold.
    fold_scopes: dict[int, dict] = {}
    for fold in fold_numbers:
        fold_dir = output_dir / "fold_results" / f"fold_{fold}"
        train_ids = fold_assignments.filter(
            (pl.col("fold") == fold) & (pl.col("split") == "train")
        ).select("row_id")
        validation_ids = fold_assignments.filter(
            (pl.col("fold") == fold) & (pl.col("split") == "validation")
        ).select("row_id")
        fold_scopes[int(fold)] = {
            "dir": fold_dir, "train_ids": train_ids,
            "validation_ids": validation_ids,
        }

    default_params = {
        "hidden_dim": int(fusion_hidden_dim),
        "market_depth": int(market_depth),
        "fusion_depth": int(fusion_depth),
        "expansion": int(residual_expansion),
        "dropout": float(fusion_dropout),
        "epochs": int(fusion_epochs),
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
    }
    if market_encoder == "tft":
        default_params["temporal_lookback"] = int(temporal_lookback)

    fold_array_cache: dict[int, tuple] = {}
    temporal_scaler_cache: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}

    def model_fit_params(params: dict) -> dict:
        """Remove preprocessing-only hyperparameters before model creation."""
        return {
            key: value for key, value in params.items()
            if key != "temporal_lookback"
        }

    def stock_groups(index: pl.DataFrame) -> np.ndarray | None:
        return _date_group_ids(index) if cross_stock_attention else None

    def fusion_model_name(variant: str) -> str:
        if cross_stock_attention:
            prefix = "timesfm_tft_cross_stock_unified_raw_text_plus"
        elif market_encoder == "tft":
            prefix = "timesfm_tft_unified_raw_text_plus"
        else:
            prefix = "timesfm_covariates_unified_raw_text_plus"
        return f"{prefix}_{variant}"

    def select_families(
        arrays: tuple, variant_families: Sequence[str]
    ) -> tuple:
        index, price, all_text_indices, placeholder, target = arrays
        selected_indices = {
            family: all_text_indices[family] for family in variant_families
        }
        return index, price, selected_indices, placeholder, target

    def fold_arrays(fold: int, variant_families: Sequence[str]):
        fold = int(fold)
        if fold not in fold_array_cache:
            print(f"Preparing and caching fold {fold} arrays")
            scope = fold_scopes[fold]
            train_arrays = _assemble_raw_fusion_arrays(
                scope["train_ids"], train_price_latents, train_targets,
                train_links, stores, families, text_fields,
            )
            val_arrays = _assemble_raw_fusion_arrays(
                scope["validation_ids"], train_price_latents, train_targets,
                train_links, stores, families, text_fields,
            )
            train_index, val_index = train_arrays[0], val_arrays[0]
            train_raw = _covariate_matrix(
                train_index, train_features, covariate_columns
            )
            val_raw = _covariate_matrix(
                val_index, train_features, covariate_columns
            )
            scaler = _fit_covariate_scaler(train_raw)
            train_sequence = train_padding = val_sequence = val_padding = None
            temporal_scaler = None
            if market_encoder == "tft":
                train_sequence, train_padding = _temporal_covariate_matrix(
                    train_index, train_features, temporal_covariate_columns,
                    max_temporal_lookback,
                )
                val_sequence, val_padding = _temporal_covariate_matrix(
                    val_index, train_features, temporal_covariate_columns,
                    max_temporal_lookback,
                )
            fold_array_cache[fold] = (
                train_arrays,
                val_arrays,
                train_raw,
                val_raw,
                scaler,
                train_sequence,
                train_padding,
                val_sequence,
                val_padding,
                temporal_scaler,
            )
        (
            train_arrays, val_arrays, train_raw, val_raw, scaler,
            train_sequence, train_padding, val_sequence, val_padding,
            temporal_scaler,
        ) = fold_array_cache[fold]
        return (
            select_families(train_arrays, variant_families),
            select_families(val_arrays, variant_families),
            train_raw,
            val_raw,
            scaler,
            train_sequence,
            train_padding,
            val_sequence,
            val_padding,
            temporal_scaler,
        )

    def scaled_temporal_view(
        scaler_scope: str,
        train_sequence: np.ndarray | None,
        train_padding: np.ndarray | None,
        val_sequence: np.ndarray | None,
        val_padding: np.ndarray | None,
        lookback: int,
    ) -> tuple:
        """Slice and fold-scale cached maximum-length TFT histories."""
        if market_encoder != "tft":
            return (
                train_sequence, train_padding, val_sequence, val_padding, None
            )
        lookback = int(lookback)
        if lookback not in lookback_candidates:
            raise ValueError(f"Unsupported TFT lookback: {lookback}")
        train_view = np.ascontiguousarray(
            train_sequence[:, -lookback:, :]
        )
        train_mask = np.ascontiguousarray(
            train_padding[:, -lookback:]
        )
        val_view = np.ascontiguousarray(
            val_sequence[:, -lookback:, :]
        )
        val_mask = np.ascontiguousarray(
            val_padding[:, -lookback:]
        )
        cache_key = (str(scaler_scope), lookback)
        if cache_key not in temporal_scaler_cache:
            temporal_scaler = _fit_temporal_scaler(train_view, train_mask)
            temporal_scaler_cache[cache_key] = temporal_scaler
        temporal_scaler = temporal_scaler_cache[cache_key]
        train_view = _apply_temporal_scaler(
            train_view, train_mask, temporal_scaler
        )
        val_view = _apply_temporal_scaler(
            val_view, val_mask, temporal_scaler
        )
        return train_view, train_mask, val_view, val_mask, temporal_scaler

    inner_split_cache: dict[
        str,
        tuple[np.ndarray, np.ndarray, dict],
    ] = {}

    def ensure_inner_split(
        scope_name: str,
        base_arrays: tuple,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        if scope_name not in inner_split_cache:
            inner_split_cache[scope_name] = _purged_inner_positions(
                base_arrays[0],
                purge_dates=forecast_horizon_weekdays,
            )
        return inner_split_cache[scope_name]

    def inner_candidate_data(
        scope_name: str,
        base_arrays: tuple,
        raw_covariates: np.ndarray,
        raw_sequence: np.ndarray | None,
        raw_padding: np.ndarray | None,
        lookback: int | None,
    ) -> tuple:
        """Prepare the purged inner split using training-only scalers."""
        (
            train_positions,
            validation_positions,
            manifest,
        ) = ensure_inner_split(
            scope_name, base_arrays
        )
        inner_train = _slice_raw_fusion_arrays(
            base_arrays, train_positions
        )
        inner_validation = _slice_raw_fusion_arrays(
            base_arrays, validation_positions
        )
        covariate_scaler = _fit_covariate_scaler(
            raw_covariates[train_positions]
        )
        inner_train_cov = _apply_covariate_scaler(
            raw_covariates[train_positions], covariate_scaler
        )
        inner_validation_cov = _apply_covariate_scaler(
            raw_covariates[validation_positions], covariate_scaler
        )
        train_sequence = train_padding = None
        validation_sequence = validation_padding = None
        temporal_scaler = None
        if market_encoder == "tft":
            (
                train_sequence,
                train_padding,
                validation_sequence,
                validation_padding,
                temporal_scaler,
            ) = scaled_temporal_view(
                f"inner_{scope_name}",
                raw_sequence[train_positions],
                raw_padding[train_positions],
                raw_sequence[validation_positions],
                raw_padding[validation_positions],
                int(lookback),
            )
        return (
            inner_train,
            inner_validation,
            inner_train_cov,
            inner_validation_cov,
            train_sequence,
            train_padding,
            validation_sequence,
            validation_padding,
            covariate_scaler,
            temporal_scaler,
            manifest,
        )

    def sampled_hyperparameters(trial) -> dict:
        params = {
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", list(hidden_dim_candidates)
            ),
            "fusion_depth": trial.suggest_int("fusion_depth", 1, 2),
            "expansion": trial.suggest_categorical("expansion", [1, 2]),
            "dropout": trial.suggest_float("dropout", 0.15, 0.50),
            "epochs": int(fusion_epochs),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 3e-4, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-4, 1e-2, log=True
            ),
            "market_depth": (
                trial.suggest_int("market_depth", 1, 2)
                if market_encoder == "mlp" else 0
            ),
        }
        if market_encoder == "tft":
            params["temporal_lookback"] = trial.suggest_categorical(
                "temporal_lookback", list(lookback_candidates)
            )
        return params

    def fit_inner_candidate(
        *,
        scope_name: str,
        base_arrays: tuple,
        raw_covariates: np.ndarray,
        raw_sequence: np.ndarray | None,
        raw_padding: np.ndarray | None,
        params: dict,
        candidate_seed: int,
        calibrate_threshold: bool = True,
    ) -> tuple[
        nn.Module,
        list[dict[str, float]],
        float,
        float,
        float,
        dict,
    ]:
        """Evaluate one configuration on the purged inner split."""
        lookback = (
            int(params["temporal_lookback"])
            if market_encoder == "tft" else None
        )
        (
            inner_train,
            inner_validation,
            inner_train_cov,
            inner_validation_cov,
            train_sequence,
            train_padding,
            validation_sequence,
            validation_padding,
            _,
            _,
            manifest,
        ) = inner_candidate_data(
            scope_name,
            base_arrays,
            raw_covariates,
            raw_sequence,
            raw_padding,
            lookback,
        )
        model, history = fit_raw_fusion_model(
            inner_train[1],
            inner_train_cov,
            inner_train[2],
            stores,
            inner_train[4],
            device,
            text_dim=raw_text_dim,
            batch_size=fusion_batch_size,
            seed=candidate_seed,
            market_encoder=market_encoder,
            market_sequence=train_sequence,
            sequence_padding_mask=train_padding,
            market_attention_heads=tft_attention_heads,
            text_attention_heads=text_attention_heads,
            text_attention_layers=text_attention_layers,
            cross_stock_attention=cross_stock_attention,
            cross_stock_attention_heads=cross_stock_attention_heads,
            stock_group_ids=stock_groups(inner_train[0]),
            validation_price=inner_validation[1],
            validation_covariates=inner_validation_cov,
            validation_text_indices=inner_validation[2],
            validation_target=inner_validation[4],
            validation_market_sequence=validation_sequence,
            validation_sequence_padding_mask=validation_padding,
            validation_stock_group_ids=stock_groups(
                inner_validation[0]
            ),
            adapter_learning_rate_multiplier=0.1,
            select_best_checkpoint=True,
            **model_fit_params(params),
        )
        selected_epoch = int(history[0]["best_epoch"])
        selected_history_row = next(
            row for row in history
            if int(row["epoch"]) == selected_epoch
        )
        optuna_objective = -float(
            selected_history_row["validation_bce"]
        )
        for row in history:
            row["calibration_best_epoch"] = float(selected_epoch)
            row["selected_epoch"] = float(selected_epoch)
        threshold = 0.5
        inner_balanced_accuracy = float(
            selected_history_row["validation_balanced_accuracy"]
        )
        if calibrate_threshold:
            score = predict_raw_fusion(
                model,
                inner_validation[1],
                inner_validation_cov,
                inner_validation[2],
                stores,
                device,
                market_encoder=market_encoder,
                market_sequence=validation_sequence,
                sequence_padding_mask=validation_padding,
                cross_stock_attention=cross_stock_attention,
                stock_group_ids=stock_groups(inner_validation[0]),
            )
            validation_truth = inner_validation[4].astype(np.int8)
            threshold = _best_validation_threshold(
                validation_truth, score
            )
            inner_balanced_accuracy = float(balanced_accuracy_score(
                validation_truth,
                (score >= threshold).astype(np.int8),
            ))
        manifest = {
            **manifest,
            "inner_best_epoch": selected_epoch,
            "selected_epoch": selected_epoch,
            "decision_threshold": float(threshold),
            "inner_balanced_accuracy": inner_balanced_accuracy,
            "threshold_calibrated": bool(calibrate_threshold),
            "optuna_objective_name": (
                "negative_validation_bce"
            ),
            "optuna_objective_value": optuna_objective,
        }
        return (
            model,
            history,
            threshold,
            inner_balanced_accuracy,
            optuna_objective,
            manifest,
        )

    def tune_inner_scope(
        *,
        scope_name: str,
        base_arrays: tuple,
        raw_covariates: np.ndarray,
        raw_sequence: np.ndarray | None,
        raw_padding: np.ndarray | None,
        study_name: str,
        study_seed: int,
    ) -> tuple[dict, float, pl.DataFrame, dict]:
        """Tune non-epoch parameters, then calibrate epoch and threshold."""
        _, _, split_manifest = ensure_inner_split(scope_name, base_arrays)
        history_dir = (
            output_dir / "inner_selection_histories" / scope_name
        )
        history_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            [split_manifest], infer_schema_length=None
        ).write_csv(
            history_dir / "inner_split.csv"
        )
        if tune_hyperparameters and tuning_trials > 0:
            try:
                import optuna
            except ImportError as error:
                raise ImportError(
                    "Hyperparameter tuning requires optuna; "
                    "install it or pass --no-tune"
                ) from error
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{(output_dir / 'optuna_study.db').resolve()}",
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=study_seed, multivariate=True
                ),
            )

            def objective(trial) -> float:
                params = sampled_hyperparameters(trial)
                (
                    model,
                    history,
                    _,
                    inner_balanced_accuracy,
                    optuna_objective,
                    _,
                ) = fit_inner_candidate(
                    scope_name=scope_name,
                    base_arrays=base_arrays,
                    raw_covariates=raw_covariates,
                    raw_sequence=raw_sequence,
                    raw_padding=raw_padding,
                    params=params,
                    # Use common random numbers so Optuna compares model
                    # configurations rather than different initializations.
                    candidate_seed=study_seed,
                    calibrate_threshold=False,
                )
                best_epoch = int(history[0]["selected_epoch"])
                trial.set_user_attr(
                    "trial_diagnostic_epoch", best_epoch
                )
                trial.set_user_attr(
                    "trial_balanced_accuracy_at_0_5",
                    float(inner_balanced_accuracy),
                )
                trial.set_user_attr(
                    "trial_validation_bce",
                    -float(optuna_objective),
                )
                pl.DataFrame(history).with_columns(
                    pl.lit(scope_name).alias("scope"),
                    pl.lit(trial.number).cast(pl.Int64).alias("trial"),
                ).write_csv(
                    history_dir / f"trial_{trial.number}.csv"
                )
                del model
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                return optuna_objective

            completed_before = sum(
                trial.state == optuna.trial.TrialState.COMPLETE
                for trial in study.trials
            )
            remaining_trials = max(
                0, int(tuning_trials) - completed_before
            )
            if remaining_trials:
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    gc_after_trial=True,
                )
            complete_trials = [
                trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            if not complete_trials:
                raise RuntimeError(
                    f"Optuna study {study_name} has no completed valid trial"
                )
            best_trial = max(
                complete_trials, key=lambda trial: float(trial.value)
            )
            winning_params = {
                **default_params,
                **best_trial.params,
                # Epoch is deliberately not inherited from the Optuna trial.
                "epochs": int(fusion_epochs),
            }
            (
                calibration_model,
                calibration_history,
                threshold,
                calibration_inner_balanced_accuracy,
                _,
                manifest,
            ) = fit_inner_candidate(
                scope_name=scope_name,
                base_arrays=base_arrays,
                raw_covariates=raw_covariates,
                raw_sequence=raw_sequence,
                raw_padding=raw_padding,
                params=winning_params,
                # Calibrate epoch/threshold with a fresh initialization after
                # Optuna has finished selecting all other hyperparameters.
                candidate_seed=study_seed + 5_000_000,
            )
            selected_epoch = int(
                calibration_history[0]["selected_epoch"]
            )
            best_params = {
                **winning_params,
                "epochs": selected_epoch,
            }
            calibration_history_frame = pl.DataFrame(
                calibration_history
            ).with_columns(
                pl.lit(scope_name).alias("scope"),
                pl.lit(best_trial.number).cast(pl.Int64).alias(
                    "optuna_best_trial"
                ),
                pl.lit("post_optuna_epoch_calibration").alias("phase"),
            )
            selected_history_path = (
                history_dir / "post_optuna_epoch_calibration.csv"
            )
            calibration_history_frame.write_csv(selected_history_path)
            del calibration_model
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            rows = [{
                "scope": scope_name,
                "trial": trial.number,
                "state": trial.state.name,
                "optuna_objective": trial.value,
                "validation_bce": trial.user_attrs.get(
                    "trial_validation_bce"
                ),
                "trial_balanced_accuracy_at_0_5": trial.user_attrs.get(
                    "trial_balanced_accuracy_at_0_5"
                ),
                "trial_diagnostic_epoch": trial.user_attrs.get(
                    "trial_diagnostic_epoch"
                ),
                "params_json": json.dumps(trial.params, sort_keys=True),
            } for trial in study.trials]
            trials = pl.DataFrame(rows, infer_schema_length=None)
            manifest = {
                **manifest,
                "optuna_best_trial": int(best_trial.number),
                "optuna_best_objective": float(best_trial.value),
                "optuna_best_validation_bce": float(
                    -best_trial.value
                ),
                "calibration_inner_balanced_accuracy": float(
                    calibration_inner_balanced_accuracy
                ),
                "epoch_selection": (
                    "post_optuna_inner_validation"
                ),
            }
            selected_trial_number = int(best_trial.number)
        else:
            (
                model,
                history,
                threshold,
                inner_balanced_accuracy,
                optuna_objective,
                manifest,
            ) = fit_inner_candidate(
                scope_name=scope_name,
                base_arrays=base_arrays,
                raw_covariates=raw_covariates,
                raw_sequence=raw_sequence,
                raw_padding=raw_padding,
                params=default_params,
                candidate_seed=study_seed,
            )
            best_params = {
                **default_params,
                "epochs": int(history[0]["selected_epoch"]),
            }
            pl.DataFrame(history).with_columns(
                pl.lit(scope_name).alias("scope"),
                pl.lit(0).cast(pl.Int64).alias("trial"),
            ).write_csv(history_dir / "fixed_default.csv")
            trials = pl.DataFrame([{
                "scope": scope_name,
                "trial": 0,
                "state": "FIXED_DEFAULT",
                "optuna_objective": optuna_objective,
                "validation_bce": -optuna_objective,
                "trial_balanced_accuracy_at_0_5": inner_balanced_accuracy,
                "trial_diagnostic_epoch": int(
                    history[0]["selected_epoch"]
                ),
                "params_json": json.dumps(default_params, sort_keys=True),
            }], infer_schema_length=None)
            del model
            selected_trial_number = 0
            selected_history_path = history_dir / "fixed_default.csv"
            manifest = {
                **manifest,
                "epoch_selection": "fixed_configuration_inner_validation",
            }
        if selected_history_path.exists():
            selected_history = pl.read_csv(selected_history_path)
            selected_history.write_csv(
                history_dir / "selected_learning_curves.csv"
            )
            _plot_inner_selection_diagnostics(
                selected_history,
                history_dir / "selected_learning_curves.png",
            )
        else:
            print(
                f"Selected inner history is unavailable for {scope_name}: "
                f"{selected_history_path}"
            )
        (history_dir / "selection.json").write_text(json.dumps({
            "scope": scope_name,
            "selected_trial": selected_trial_number,
            "epoch_selection": manifest["epoch_selection"],
            "hyperparameters": best_params,
            "selected_epoch": int(best_params["epochs"]),
            "decision_threshold": float(threshold),
            "inner_split": {
                key: (
                    str(value)
                    if key.endswith(("_end", "_start")) else value
                )
                for key, value in manifest.items()
            },
        }, indent=2, sort_keys=True))
        return best_params, threshold, trials, manifest

    # True nested selection: each outer fold owns an inner purged split.
    tuning_trial_parts: list[pl.DataFrame] = []
    nested_selection_rows: list[dict] = []
    outer_best_params: dict[int, dict] = {}
    prediction_parts = []
    diagnostic_histories: dict[str, list[pl.DataFrame]] = {}
    for fold in fold_numbers:
        fold = int(fold)
        (
            tuning_train,
            _,
            tuning_train_raw,
            _,
            _,
            tuning_train_sequence,
            tuning_train_padding,
            _,
            _,
            _,
        ) = fold_arrays(fold, variants[tuning_variant])
        fold_params, tuning_threshold, trials, split_manifest = tune_inner_scope(
            scope_name=f"outer_fold_{fold}",
            base_arrays=tuning_train,
            raw_covariates=tuning_train_raw,
            raw_sequence=tuning_train_sequence,
            raw_padding=tuning_train_padding,
            study_name=(
                f"timesfm_{market_encoder}_nested_outer_fold_{fold}_"
                f"{study_signature}_v7"
            ),
            study_seed=seed + 10_000 * fold,
        )
        outer_best_params[fold] = fold_params
        tuning_trial_parts.append(
            trials.with_columns(
                pl.lit(fold).cast(pl.Int8).alias("outer_fold")
            )
        )

        for variant, variant_families in variants.items():
            (
                train_a,
                val_a,
                train_raw,
                val_raw,
                _,
                train_sequence,
                train_padding,
                val_sequence,
                val_padding,
                _,
            ) = fold_arrays(fold, variant_families)
            variant_params = dict(fold_params)
            threshold = float(tuning_threshold)
            selection_manifest = split_manifest

            # Non-tuning text variants share the selected architecture but get
            # their own inner-selected epoch and probability threshold.
            if variant != tuning_variant:
                variant_selection_params = {
                    **variant_params,
                    "epochs": int(fusion_epochs),
                }
                (
                    inner_model,
                    inner_history,
                    threshold,
                    _,
                    _,
                    selection_manifest,
                ) = fit_inner_candidate(
                    scope_name=f"outer_fold_{fold}",
                    base_arrays=train_a,
                    raw_covariates=train_raw,
                    raw_sequence=train_sequence,
                    raw_padding=train_padding,
                    params=variant_selection_params,
                    candidate_seed=seed + 20_000 * fold,
                )
                variant_params["epochs"] = int(
                    inner_history[0]["selected_epoch"]
                )
                pl.DataFrame(inner_history).with_columns(
                    pl.lit(f"outer_fold_{fold}").alias("scope"),
                    pl.lit(variant).alias("variant"),
                ).write_csv(
                    output_dir / "inner_selection_histories" /
                    f"outer_fold_{fold}" /
                    f"variant_{variant}_selection.csv"
                )
                del inner_model

            scaler = _fit_covariate_scaler(train_raw)
            train_cov = _apply_covariate_scaler(train_raw, scaler)
            val_cov = _apply_covariate_scaler(val_raw, scaler)
            temporal_scaler = None
            selected_temporal_lookback = (
                int(variant_params["temporal_lookback"])
                if market_encoder == "tft" else None
            )
            if market_encoder == "tft":
                (
                    train_sequence,
                    train_padding,
                    val_sequence,
                    val_padding,
                    temporal_scaler,
                ) = scaled_temporal_view(
                    f"outer_refit_{fold}",
                    train_sequence,
                    train_padding,
                    val_sequence,
                    val_padding,
                    selected_temporal_lookback,
                )

            model_name = fusion_model_name(variant)
            config_id = f"{model_name}__nested"
            model_path = (
                fold_scopes[fold]["dir"] / "fusion_models" /
                f"{model_name}.pt"
            )
            model, history = fit_raw_fusion_model(
                train_a[1],
                train_cov,
                train_a[2],
                stores,
                train_a[4],
                device,
                text_dim=raw_text_dim,
                batch_size=fusion_batch_size,
                seed=seed + fold,
                market_encoder=market_encoder,
                market_sequence=train_sequence,
                sequence_padding_mask=train_padding,
                market_attention_heads=tft_attention_heads,
                text_attention_heads=text_attention_heads,
                text_attention_layers=text_attention_layers,
                cross_stock_attention=cross_stock_attention,
                cross_stock_attention_heads=cross_stock_attention_heads,
                stock_group_ids=stock_groups(train_a[0]),
                validation_price=val_a[1],
                validation_covariates=val_cov,
                validation_text_indices=val_a[2],
                validation_target=val_a[4],
                validation_market_sequence=val_sequence,
                validation_sequence_padding_mask=val_padding,
                validation_stock_group_ids=stock_groups(val_a[0]),
                select_best_checkpoint=False,
                **model_fit_params(variant_params),
            )
            save_torch_model(model_path, model, {
                "scope": f"fold_{fold}",
                "model": model_name,
                "selection": "nested_inner_validation",
                "price_encoder": "timesfm",
                "price_encoder_frozen": True,
                "market_encoder": market_encoder,
                "covariates": list(covariate_columns),
                "covariate_preprocessing": "outer-training median then zscore",
                "text_families": list(variant_families),
                "text_fields": text_fields,
                "raw_embedding_dims": {
                    family: stores[family].input_dim
                    for family in variant_families
                },
                "raw_text_shared_dim": raw_text_dim,
                "text_attention_heads": text_attention_heads,
                "text_attention_layers": text_attention_layers,
                "adapter_learning_rate_multiplier": 0.1,
                "cross_stock_attention": cross_stock_attention,
                "cross_stock_attention_heads": (
                    cross_stock_attention_heads
                    if cross_stock_attention else None
                ),
                "temporal_covariates": list(temporal_covariate_columns),
                "temporal_lookback": selected_temporal_lookback,
                "decision_threshold": threshold,
                "hyperparameters": variant_params,
                "inner_split": {
                    key: str(value) if key.endswith(("_end", "_start")) else value
                    for key, value in selection_manifest.items()
                },
            })
            _save_covariate_scaler(
                fold_scopes[fold]["dir"] / "fusion_models" /
                f"{model_name}_covariate_scaler.csv",
                covariate_columns,
                scaler,
            )
            if temporal_scaler is not None:
                _save_covariate_scaler(
                    fold_scopes[fold]["dir"] / "fusion_models" /
                    f"{model_name}_temporal_scaler.csv",
                    temporal_covariate_columns,
                    temporal_scaler,
                )

            history_frame = pl.DataFrame(history).with_columns(
                pl.lit(fold).cast(pl.Int8).alias("fold"),
                pl.lit(model_name).alias("model"),
                pl.lit(variant).alias("variant"),
            )
            history_frame.write_csv(
                fold_scopes[fold]["dir"] / "fusion_models" /
                f"{model_name}_training_history.csv"
            )
            diagnostic_histories.setdefault(model_name, []).append(
                history_frame
            )
            score = predict_raw_fusion(
                model,
                val_a[1],
                val_cov,
                val_a[2],
                stores,
                device,
                market_encoder=market_encoder,
                market_sequence=val_sequence,
                sequence_padding_mask=val_padding,
                cross_stock_attention=cross_stock_attention,
                stock_group_ids=stock_groups(val_a[0]),
            )
            prediction_parts.append(val_a[0].with_columns(
                pl.lit(feature_set).alias("feature_set"),
                pl.lit(model_name).alias("model"),
                pl.lit(config_id).alias("config_id"),
                pl.lit(fold).cast(pl.Int8).alias("fold"),
                pl.col("target_up").cast(pl.Int8).alias("y_true"),
                pl.Series("y_score", score.astype(np.float32)),
                pl.lit(threshold).alias("decision_threshold"),
            ).drop("target_up"))
            nested_selection_rows.append({
                "outer_fold": fold,
                "variant": variant,
                "model": model_name,
                "best_epoch": int(variant_params["epochs"]),
                "decision_threshold": threshold,
                "inner_best_epoch": selection_manifest[
                    "inner_best_epoch"
                ],
                "inner_train_rows": selection_manifest["inner_train_rows"],
                "inner_validation_rows": selection_manifest[
                    "inner_validation_rows"
                ],
                "purged_dates": selection_manifest["purged_dates"],
                "inner_train_end": selection_manifest["inner_train_end"],
                "inner_validation_start": selection_manifest[
                    "inner_validation_start"
                ],
                "inner_validation_end": selection_manifest[
                    "inner_validation_end"
                ],
                "params_json": json.dumps(
                    variant_params, sort_keys=True
                ),
            })
            del model, train_a, val_a, train_cov, val_cov
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    if tuning_trial_parts:
        tuning_trials_table = pl.concat(
            tuning_trial_parts, how="diagonal_relaxed"
        ).sort(["outer_fold", "trial"])
    else:
        tuning_trials_table = pl.DataFrame(schema={
            "scope": pl.String,
            "trial": pl.Int64,
            "state": pl.String,
            "optuna_objective": pl.Float64,
            "validation_bce": pl.Float64,
            "trial_balanced_accuracy_at_0_5": pl.Float64,
            "trial_diagnostic_epoch": pl.Int64,
            "params_json": pl.String,
            "outer_fold": pl.Int8,
        })
    tuning_trials_table.write_csv(
        output_dir / "nested_optuna_trials.csv"
    )
    if nested_selection_rows:
        nested_selection = pl.DataFrame(
            nested_selection_rows, infer_schema_length=None
        ).sort(["outer_fold", "variant"])
    else:
        nested_selection = pl.DataFrame(schema={
            "outer_fold": pl.Int8,
            "variant": pl.String,
            "model": pl.String,
            "best_epoch": pl.Int64,
            "decision_threshold": pl.Float64,
            "inner_best_epoch": pl.Int64,
            "inner_train_rows": pl.Int64,
            "inner_validation_rows": pl.Int64,
            "purged_dates": pl.Int64,
            "inner_train_end": pl.Date,
            "inner_validation_start": pl.Date,
            "inner_validation_end": pl.Date,
            "params_json": pl.String,
        })
    nested_selection.write_csv(
        output_dir / "nested_outer_fold_selection.csv"
    )
    (output_dir / "outer_fold_hyperparameters.json").write_text(
        json.dumps({
            str(fold): params
            for fold, params in outer_best_params.items()
        }, indent=2, sort_keys=True)
    )

    diagnostic_dir = output_dir / "training_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    combined_diagnostics = []
    for model_name, history_parts in diagnostic_histories.items():
        model_history = pl.concat(history_parts).sort(["fold", "epoch"])
        model_history.write_csv(
            diagnostic_dir / f"{model_name}_fold_learning_curves.csv"
        )
        _plot_fold_training_diagnostics(
            model_history,
            diagnostic_dir / f"{model_name}_fold_learning_curves.png",
        )
        combined_diagnostics.append(model_history)
    if combined_diagnostics:
        training_diagnostics = pl.concat(combined_diagnostics).sort(
            ["model", "fold", "epoch"]
        )
    else:
        training_diagnostics = pl.DataFrame(schema={
            "epoch": pl.Float64,
            "bce": pl.Float64,
            "train_bce": pl.Float64,
            "train_positive_rate": pl.Float64,
            "validation_bce": pl.Float64,
            "validation_accuracy": pl.Float64,
            "validation_balanced_accuracy": pl.Float64,
            "validation_roc_auc": pl.Float64,
            "validation_optimized_balanced_accuracy": pl.Float64,
            "validation_optimized_threshold": pl.Float64,
            "validation_mean_probability": pl.Float64,
            "validation_positive_rate": pl.Float64,
            "best_epoch": pl.Float64,
            "is_best_epoch": pl.Float64,
            "stopped_early": pl.Float64,
            "fold": pl.Int8,
            "model": pl.String,
            "variant": pl.String,
        })
    training_diagnostics.write_csv(
        diagnostic_dir / "all_fold_learning_curves.csv"
    )

    if prediction_parts:
        raw_oof = pl.concat(prediction_parts)
    else:
        raw_oof = pl.DataFrame(schema={
            "row_id": pl.UInt64,
            "date": pl.Date,
            "ticker": pl.String,
            "feature_set": pl.String,
            "model": pl.String,
            "config_id": pl.String,
            "fold": pl.Int8,
            "y_true": pl.Int8,
            "y_score": pl.Float32,
            "decision_threshold": pl.Float64,
        })
    threshold_table = raw_oof.select([
        "model", "fold", "decision_threshold",
    ]).unique().sort(["model", "fold"])
    threshold_table.write_csv(output_dir / "selected_decision_thresholds.csv")
    oof_predictions = (
        raw_oof.with_columns(
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
        train_rows = fold_arrays(
            int(first["fold"]), variants[tuning_variant]
        )[0][0].height
        metric_rows.append(_metric_row(
            feature_set, first["model"], first["config_id"],
            json.dumps({"selection": "nested_inner_validation"}),
            int(first["fold"]), train_rows, partition["y_true"].to_numpy(),
            partition["y_score"].to_numpy(), float(first["decision_threshold"]),
        ))
    if metric_rows:
        fold_metrics = pl.DataFrame(metric_rows).sort(["model", "fold"])
    else:
        fold_metrics = pl.DataFrame(schema={
            "feature_set": pl.String,
            "model": pl.String,
            "config_id": pl.String,
            "params_json": pl.String,
            "decision_threshold": pl.Float64,
            "fold": pl.Int8,
            "train_rows_used": pl.Int64,
            "validation_rows": pl.Int64,
            "accuracy": pl.Float64,
            "balanced_accuracy": pl.Float64,
            "roc_auc": pl.Float64,
            "brier_score": pl.Float64,
        })
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

    # Fit fresh raw-family adapters, scalers, and networks on all labeled rows.
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
    all_train_ids = final_targets.select("row_id")
    all_test_ids = scoped_test_price.select("row_id")

    # Repeat inner-only selection on the complete labeled training set. The
    # selected epoch is then used for a fresh refit that includes every row.
    final_tuning_arrays = _assemble_raw_fusion_arrays(
        all_train_ids,
        train_price_latents,
        train_targets,
        train_links,
        stores,
        variants[tuning_variant],
        text_fields,
    )
    final_tuning_raw = _covariate_matrix(
        final_tuning_arrays[0], train_features, covariate_columns
    )
    final_tuning_sequence = final_tuning_padding = None
    if market_encoder == "tft":
        (
            final_tuning_sequence,
            final_tuning_padding,
        ) = _temporal_covariate_matrix(
            final_tuning_arrays[0],
            train_features,
            temporal_covariate_columns,
            max_temporal_lookback,
        )
    (
        final_best_params,
        final_tuning_threshold,
        final_trials,
        final_split_manifest,
    ) = tune_inner_scope(
        scope_name="final_full_training",
        base_arrays=final_tuning_arrays,
        raw_covariates=final_tuning_raw,
        raw_sequence=final_tuning_sequence,
        raw_padding=final_tuning_padding,
        study_name=(
            f"timesfm_{market_encoder}_nested_final_"
            f"{study_signature}_v7"
        ),
        study_seed=seed + 900_000,
    )
    final_trials = final_trials.with_columns(
        pl.lit(None).cast(pl.Int8).alias("outer_fold")
    )
    tuning_trials_table = pl.concat(
        [tuning_trials_table, final_trials],
        how="diagonal_relaxed",
    )
    tuning_trials_table.write_csv(
        output_dir / "nested_optuna_trials.csv"
    )
    (output_dir / "best_hyperparameters.json").write_text(json.dumps({
        "selection": "nested_inner_validation",
        "epoch_selection": final_split_manifest["epoch_selection"],
        "optuna_study_signature": study_signature,
        "cross_stock_attention": bool(cross_stock_attention),
        "cross_stock_attention_heads": (
            cross_stock_attention_heads
            if cross_stock_attention else None
        ),
        "training_mode": (
            "nested-folds" if run_outer_folds else "full-only"
        ),
        "outer_fold_hyperparameters": {
            str(fold): params
            for fold, params in outer_best_params.items()
        },
        "final_hyperparameters": final_best_params,
        "final_decision_threshold": final_tuning_threshold,
        "final_inner_split": {
            key: str(value) if key.endswith(("_end", "_start")) else value
            for key, value in final_split_manifest.items()
        },
        "test_data_used_for_selection": False,
    }, indent=2, sort_keys=True))

    final_prediction_parts = []
    final_thresholds: dict[str, float] = {}
    for variant, variant_families in variants.items():
        train_a = _assemble_raw_fusion_arrays(
            all_train_ids, train_price_latents, train_targets,
            train_links, stores, variant_families, text_fields,
        )
        test_a = _assemble_raw_fusion_arrays(
            all_test_ids, scoped_test_price, scoped_test_targets,
            scoped_test_links, stores, variant_families, text_fields,
            require_target=False,
        )
        train_raw = _covariate_matrix(train_a[0], train_features, covariate_columns)
        test_raw = _covariate_matrix(test_a[0], scoped_test_features, covariate_columns)
        variant_params = dict(final_best_params)
        threshold = float(final_tuning_threshold)
        variant_selection_manifest = final_split_manifest
        raw_train_sequence = raw_train_padding = None
        if market_encoder == "tft":
            raw_train_sequence, raw_train_padding = _temporal_covariate_matrix(
                train_a[0],
                train_features,
                temporal_covariate_columns,
                max_temporal_lookback,
            )
        if variant != tuning_variant:
            variant_selection_params = {
                **variant_params,
                "epochs": int(fusion_epochs),
            }
            (
                inner_model,
                inner_history,
                threshold,
                _,
                _,
                variant_selection_manifest,
            ) = fit_inner_candidate(
                scope_name="final_full_training",
                base_arrays=train_a,
                raw_covariates=train_raw,
                raw_sequence=raw_train_sequence,
                raw_padding=raw_train_padding,
                params=variant_selection_params,
                candidate_seed=seed + 910_000,
            )
            variant_params["epochs"] = int(
                inner_history[0]["selected_epoch"]
            )
            pl.DataFrame(inner_history).with_columns(
                pl.lit("final_full_training").alias("scope"),
                pl.lit(variant).alias("variant"),
            ).write_csv(
                output_dir / "inner_selection_histories" /
                "final_full_training" /
                f"variant_{variant}_selection.csv"
            )
            del inner_model
        scaler = _fit_covariate_scaler(train_raw)
        train_cov = _apply_covariate_scaler(train_raw, scaler)
        test_cov = _apply_covariate_scaler(test_raw, scaler)
        train_sequence = train_padding = test_sequence = test_padding = None
        temporal_scaler = None
        selected_temporal_lookback = (
            int(variant_params["temporal_lookback"])
            if market_encoder == "tft" else None
        )
        if market_encoder == "tft":
            train_sequence, train_padding = _temporal_covariate_matrix(
                train_a[0], train_features, temporal_covariate_columns,
                selected_temporal_lookback,
            )
            test_sequence, test_padding = _temporal_covariate_matrix(
                test_a[0], scoped_test_features, temporal_covariate_columns,
                selected_temporal_lookback,
            )
            temporal_scaler = _fit_temporal_scaler(
                train_sequence, train_padding
            )
            train_sequence = _apply_temporal_scaler(
                train_sequence, train_padding, temporal_scaler
            )
            test_sequence = _apply_temporal_scaler(
                test_sequence, test_padding, temporal_scaler
            )
        model_name = fusion_model_name(variant)
        model, history = fit_raw_fusion_model(
            train_a[1], train_cov, train_a[2], stores, train_a[4],
            device, text_dim=raw_text_dim,
            batch_size=fusion_batch_size, seed=seed,
            market_encoder=market_encoder,
            market_sequence=train_sequence,
            sequence_padding_mask=train_padding,
            market_attention_heads=tft_attention_heads,
            text_attention_heads=text_attention_heads,
            text_attention_layers=text_attention_layers,
            cross_stock_attention=cross_stock_attention,
            cross_stock_attention_heads=cross_stock_attention_heads,
            stock_group_ids=stock_groups(train_a[0]),
            **model_fit_params(variant_params),
        )
        save_torch_model(final_dir / "fusion_models" / f"{model_name}.pt", model, {
            "scope": "final", "model": model_name,
            "price_encoder": "timesfm", "price_encoder_frozen": True,
            "market_encoder": market_encoder,
            "covariates": list(covariate_columns),
            "covariate_preprocessing": "all-training median then zscore",
            "text_families": list(variant_families),
            "text_fields": text_fields,
            "raw_embedding_dims": {
                family: stores[family].input_dim
                for family in variant_families
            },
            "raw_text_shared_dim": raw_text_dim,
            "text_attention_heads": text_attention_heads,
            "text_attention_layers": text_attention_layers,
            "adapter_learning_rate_multiplier": 0.1,
            "cross_stock_attention": cross_stock_attention,
            "cross_stock_attention_heads": (
                cross_stock_attention_heads
                if cross_stock_attention else None
            ),
            "architecture": (
                "tft_plus_raw_text_attention_plus_cross_stock_attention"
                if cross_stock_attention else
                "tft_plus_raw_family_adapters_self_and_cross_attention"
                if market_encoder == "tft" else
                "market_mlp_plus_raw_family_adapters_self_and_cross_attention"
            ),
            "temporal_covariates": list(temporal_covariate_columns),
            "temporal_lookback": selected_temporal_lookback,
            "decision_threshold": threshold,
            "hyperparameters": variant_params,
            "inner_split": {
                key: (
                    str(value)
                    if key.endswith(("_end", "_start")) else value
                )
                for key, value in variant_selection_manifest.items()
            },
        })
        _save_covariate_scaler(
            final_dir / "fusion_models" / f"{model_name}_covariate_scaler.csv",
            covariate_columns, scaler,
        )
        if temporal_scaler is not None:
            _save_covariate_scaler(
                final_dir / "fusion_models" /
                f"{model_name}_temporal_scaler.csv",
                temporal_covariate_columns,
                temporal_scaler,
            )
        pl.DataFrame(history).write_csv(
            final_dir / "fusion_models" / f"{model_name}_training_history.csv"
        )
        score = predict_raw_fusion(
            model, test_a[1], test_cov, test_a[2], stores, device,
            market_encoder=market_encoder,
            market_sequence=test_sequence,
            sequence_padding_mask=test_padding,
            cross_stock_attention=cross_stock_attention,
            stock_group_ids=stock_groups(test_a[0]),
        )
        final_thresholds[model_name] = threshold
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
                "decision_threshold": final_thresholds[model_name],
            }
            if scored.height:
                row.update(directional_classification_metrics(
                    scored["y_true"].to_numpy(), scored["y_score"].to_numpy(),
                    final_thresholds[model_name],
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
            "decision_threshold": final_thresholds[model_name],
        })
    submission_manifest = pl.DataFrame(manifest_rows, infer_schema_length=None)
    submission_manifest.write_csv(output_dir / "submission_manifest.csv")

    if run_outer_folds:
        baseline_tables = [
            pl.read_csv(path)
            for path in sorted(
                baseline_dir.glob("*/selected_config_fold_metrics_*.csv")
            )
        ]
        comparison_folds = pl.concat(
            [*baseline_tables, fold_metrics], how="diagonal_relaxed"
        ) if baseline_tables else fold_metrics
        comparison_aggregate = _aggregate_fold_metrics(
            comparison_folds.select(fold_metrics.columns)
        )
    else:
        # There is no unbiased outer-fold estimate to compare in full-only
        # mode; keep these artifacts explicitly empty.
        comparison_folds = fold_metrics
        comparison_aggregate = aggregate
    comparison_folds.write_csv(output_dir / "comparison_with_baselines_fold_metrics.csv")
    comparison_aggregate.write_csv(output_dir / "comparison_with_baselines_aggregate.csv")
    return {
        "tuning_trials": tuning_trials_table,
        "nested_selection": nested_selection,
        "training_diagnostics": training_diagnostics,
        "fold_metrics": fold_metrics, "aggregate": aggregate,
        "oof_predictions": oof_predictions,
        "final_predictions": final_predictions, "final_metrics": final_metrics,
        "submission_manifest": submission_manifest,
        "comparison_folds": comparison_folds,
        "comparison_aggregate": comparison_aggregate,
    }
