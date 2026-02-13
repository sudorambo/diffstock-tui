# Changelog

All notable changes to DiffStock-TUI are documented here.

## [0.1.1] - 2026-02-13

### Added
- **Early stopping**: Training halts when validation loss stops improving. Configurable via `--patience <N>` CLI flag (default: 20 epochs).
- **Missing weights warning**: `tracing::warn!` is now emitted when `model_weights.safetensors` is not found, clearly stating predictions will be meaningless.
- **Inference tests**: Two new tests in `inference.rs`:
  - `test_inference_with_mock_data` — trains 1 epoch, runs inference, verifies output structure and percentile ordering (p10 <= p50 <= p90).
  - `test_inference_without_weights` — verifies the zeros fallback runs without panicking.
- **Centralized model hyperparameters**: `INPUT_DIM`, `HIDDEN_DIM`, `NUM_LAYERS`, `DIFF_STEPS`, and `PATIENCE` constants added to `config.rs`.

### Fixed
- **Timestep OOB bug**: `Tensor::rand(0.0, 100.0, ...).floor()` could produce index 100 into a 100-element tensor. Added `.clamp(0.0, 99.0)` in both training and validation loops.
- **Validation `to_vec1` roundtrip**: Replaced the unnecessary `to_vec1::<f32>` -> `map` -> `Tensor::new` conversion in the validation loop with a direct `to_dtype(DType::U32)` cast (matching the training loop).
- **Dead variable**: Removed unused `_input_dim` binding in `train.rs`.
- **Train/inference hyperparameter drift**: `input_dim`, `hidden_dim`, `num_layers`, and `diff_steps` were hardcoded independently in both `train.rs` and `inference.rs`. Now both import from `config.rs`, eliminating the risk of silent weight shape mismatches.

## [0.1.0] - Initial Release

### Added
- TimeGrad-inspired conditional diffusion model (LSTM Encoder + WaveNet Denoiser).
- Multi-asset training with 18 ETF/stock tickers and asset ID embeddings.
- Probabilistic inference with P10/P30/P50/P70/P90 percentile cones via Monte Carlo sampling.
- Terminal UI (`ratatui`) and GUI (`egui`) interfaces.
- Yahoo Finance data fetching with retry logic and local JSON caching.
- Model checkpointing (saves best weights by validation loss).
- CUDA GPU support behind `cuda` cargo feature flag with `--cuda` CLI flag.
- Backtesting mode for SPY with coverage probability reporting.
- Configurable hyperparameters via CLI (`--epochs`, `--batch-size`, `--learning-rate`).
