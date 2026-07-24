# Multi-modal time-series forecasting

## Frozen price backbones

The Stage 2 entry points support two frozen price encoders:

- `timesfm` (default): the existing univariate `ret_20` TimesFM 2.5 path.
- `chronos2`: a joint multivariate Chronos-2 context containing, in order,
  `ret_1`, `ret_5`, `ret_20`, `log_hl_range`, `log_close_open`, `log_volume`,
  and `volume_change_1`.

Chronos-2 is never fine-tuned. Its official `embed` interface returns encoder
states with shape `(n_variates, num_patches + 2, d_model)`, ordered as context
patches followed by `[REG]` and a masked-output patch. The implementation
removes those final two special tokens, mean-pools the observed context-patch
states separately for each variate, and concatenates the per-variate vectors.
This retains variable identity and lets the existing market/text fusion heads
consume the frozen representation.

Install a Chronos version that provides the Chronos-2 embedding interface:

```bash
python -m pip install "chronos-forecasting>=2.1.0"
```

Run the residual fusion model:

```bash
python stage2_pretrained_forecasts.py --price-encoder chronos2
```

Run either TFT variant:

```bash
python stage2_tft_forecasts.py --price-encoder chronos2
python stage2_cross_stock_tft_forecasts.py --price-encoder chronos2
```

Use `--no-price-extraction` to require an existing compatible cache or
`--force-price-refresh` to rebuild it. Chronos-2 caches have a sidecar manifest
containing the model ID, ordered inputs, context settings, and pooling rule so
that an incompatible representation is not silently reused.
