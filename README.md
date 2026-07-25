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

## Chronos-2 LoRA or TopLoRA with text fusion

The adapter runner differs from the frozen paths above: it calls the
differentiable Chronos-2 encoder directly, freezes the pretrained parameters,
and jointly trains LoRA or TopLoRA adapters in the Chronos attention projections
with the existing raw-text adapters, text self-attention, market-conditioned
text cross-attention, fusion layers, and directional classifier. The pretrained
Chronos forecasting head is not used because the downstream target is a binary
direction decision.

Install the optional fine-tuning dependencies:

```bash
python -m pip install "chronos-forecasting>=2.1.0" "peft>=0.18.1"
```

Run the separate pipeline:

```bash
python stage2_chronos2_lora_forecasts.py
```

Run the token-wise TopLoRA variant:

```bash
python stage2_chronos2_lora_forecasts.py --adapter-type toplora
```

TopLoRA replaces the shared low-rank update \(BA\) with the token-conditioned
update \(B\Sigma_XA\), where
\(\Sigma_X=\operatorname{Diag}(\exp(\operatorname{RMSNorm}(\Theta X)))\).
For Chronos-2, each time-series patch embedding is the corresponding token.
LoRA and TopLoRA write to separate artifact directories.

Early stopping is shared across the residual, TFT, cross-stock TFT, LoRA, and
TopLoRA implementations. Configure its minimum training duration and patience
in `model_config.py` with `EARLY_STOPPING_MIN_EPOCHS` and
`EARLY_STOPPING_PATIENCE`; `EARLY_STOPPING_MIN_DELTA` controls the improvement
threshold. Checkpoints before the configured minimum are not eligible for
selection, so every selected checkpoint and final refit runs for at least the
minimum number of epochs.

Decision thresholds are configured globally in `model_config.py`. Set
`CALIBRATE_DECISION_THRESHOLD = False` and
`FIXED_DECISION_THRESHOLD = 0.5` to classify directly at probability 0.5;
set calibration to `True` to select a threshold on purged validation data.

The runner performs purged inner epoch selection and applies the configured
fixed or calibrated threshold policy, then initializes a fresh adapter and
refits on all labeled training rows. It saves the compact Chronos adapter
separately from the multimodal head and writes test metrics, predictions, and
the final submission under the output directory selected for the adapter type.
