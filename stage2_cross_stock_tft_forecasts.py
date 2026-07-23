"""Run Stage 2 TFT fusion with cross-sectional attention among stocks.

The per-stock TimesFM, temporal-market, and raw-text encoders are identical to
``stage2_tft_forecasts.py``. After those representations are fused, stocks from
the same forecast date attend to one another before the directional classifier.
Attention never crosses forecast dates.
"""

from __future__ import annotations

from stage2_tft_forecasts import (
    ARTIFACT_DIR,
    parse_args,
    run_tft_pipeline,
)


OUTPUT_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_temporal_tft_cross_stock_unified_raw_text_attention"
)
CROSS_STOCK_ATTENTION_HEADS = 4


def main() -> None:
    args = parse_args(description=__doc__)
    run_tft_pipeline(
        args,
        output_dir=OUTPUT_DIR,
        cross_stock_attention=True,
        cross_stock_attention_heads=CROSS_STOCK_ATTENTION_HEADS,
    )


if __name__ == "__main__":
    main()
