"""Run Stage 2 TFT fusion with cross-sectional attention among stocks.

The per-stock TimesFM, temporal-market, and raw-text encoders are identical to
``stage2_tft_forecasts.py``. After those representations are fused, stocks from
the same forecast date attend to one another before the directional classifier.
Attention never crosses forecast dates.
"""

from __future__ import annotations

from model_config import (
    CROSS_STOCK_ATTENTION_HEADS,
    CROSS_STOCK_TFT_OUTPUT_DIR as OUTPUT_DIR,
)
from stage2_tft_forecasts import parse_args, run_tft_pipeline


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
