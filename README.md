# DiffStock-TUI

**DiffStock-TUI** is a Rust-based TUI/GUI application for probabilistic stock price forecasting using a **Conditional Diffusion Model**. It uses Hugging Face's `candle` framework to generate high-fidelity future price paths.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Rust](https://img.shields.io/badge/rust-2024-orange.svg)
![AI](https://img.shields.io/badge/AI-Diffusion%20Model-purple.svg)

## Features

*   **Generative AI**: TimeGrad-inspired diffusion model (LSTM Encoder + WaveNet Denoiser).
*   **Multi-Asset Training**: Learns asset-specific patterns using ID embeddings.
*   **Robust Training**: Includes validation sets, model checkpointing (saves best weights), early stopping, and configurable hyperparameters.
*   **Probabilistic Inference**: Generates P10-P90 confidence intervals via 500+ Monte Carlo paths.
*   **Efficient Data**: Local caching (`.cache/`) and retry logic for Yahoo Finance API.
*   **Dual Interface**: Keyboard-driven TUI (`ratatui`) and interactive GUI (`egui`).

## Requirements

*   **Rust**: Stable or Nightly (2024 edition support required).
*   **CUDA Toolkit** (optional): Required only for GPU acceleration via the `cuda` feature.

## Installation

```bash
git clone https://github.com/sudorambo/diffstock-tui.git
cd diffstock-tui

# Build for CPU (default)
cargo build --release

# Build with CUDA GPU support (requires CUDA toolkit)
cargo build --release --features cuda
```

## Usage

### 1. Training
Train the model on 5 years of historical data (SPY, QQQ, etc.). The model automatically splits data (80/20) and saves the best weights to `model_weights.safetensors`.

**Default Training:**
```bash
cargo run --release -- --train
```

**Custom Hyperparameters:**
```bash
cargo run --release -- --train --epochs 100 --batch-size 32 --learning-rate 0.0005 --patience 30
```

**Train on GPU:**
```bash
cargo run --release --features cuda -- --train --cuda
```
Training stops early if validation loss doesn't improve for `--patience` epochs (default: 20). Configuration (tickers, defaults) can be edited in `src/config.rs`.

### 2. Forecasting
Run the interactive interface to visualize forecasts.

**Terminal UI (Default):**
```bash
cargo run --release
```
*   **Controls**: Type ticker -> `Enter` to fetch, `r` to reset, `q` to quit.

**Graphical UI:**
```bash
cargo run --release -- --gui
```

Add `--cuda` to any command above to use GPU acceleration (requires the `cuda` feature at compile time). If CUDA is unavailable at runtime, it falls back to CPU automatically.

### 3. Backtesting
Validate performance on historical SPY data.
```bash
cargo run --release -- --backtest
```

## Technical Architecture

*   **ML Framework**: `candle-core` & `candle-nn` (Rust-native).
*   **Model**: Conditional Gaussian Diffusion with LSTM context and Asset Embeddings.
*   **Backend**: `tokio` async runtime with `tracing` for structured logging.
*   **Frontend**: `ratatui` (TUI) and `egui` (GUI).

## Disclaimer

**Educational use only.** Do not use these forecasts for financial trading.

## License

MIT License - see [LICENSE](LICENSE).
