"""Run Stage 2 TFT fusion with cross-sectional attention among stocks.

The optional per-stock frozen TSFM, temporal-market, and raw-text encoders are
identical to ``stage2_tft_forecasts.py``. After those representations are
fused, stocks from the same forecast date attend to one another before the
directional classifier. Attention never crosses forecast dates.
"""

from __future__ import annotations

from model_config import (
    CROSS_STOCK_ATTENTION_HEADS,
    CROSS_STOCK_TFT_OUTPUT_DIRS,
    USE_PRETRAINED_PRICE_MODEL,
)
from stage2_tft_forecasts import parse_args, run_tft_pipeline


def _cross_stock_output_dir(args):
    pretrained_output = CROSS_STOCK_TFT_OUTPUT_DIRS[args.price_encoder]
    no_pretrained_model = bool(
        getattr(
            args,
            "no_pretrained_model",
            not USE_PRETRAINED_PRICE_MODEL,
        )
    )
    if no_pretrained_model:
        return (
            pretrained_output.parent
            / "tft_cross_stock_no_pretrained_price_"
            "unified_raw_text_attention"
        )
    return pretrained_output


def main() -> None:
    args = parse_args(description=__doc__)
    run_tft_pipeline(
        args,
        output_dir=_cross_stock_output_dir(args),
        cross_stock_attention=True,
        cross_stock_attention_heads=CROSS_STOCK_ATTENTION_HEADS,
    )


if __name__ == "__main__":
    main()
