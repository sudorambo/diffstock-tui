# Improvement Plan for DiffStock-TUI

## 1. Configuration Management
**Goal**: Remove hardcoded values and allow flexible configuration.
- [x] Create a `config.toml` or `src/config.rs` to store:
    - Training tickers (currently hardcoded in `src/train.rs`).
    - Hyperparameters: `LOOKBACK`, `FORECAST`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`.
    - API endpoints/timeouts.
- [x] Update `src/train.rs` and `src/main.rs` to use this configuration.
- [x] Add CLI arguments to override config values (e.g., `--epochs 100`).

## 2. Error Handling & Logging
**Goal**: Improve observability and robustness.
- [x] Replace `println!` / `eprintln!` with `tracing` or `log` crate.
- [x] Implement retry logic for `fetch_range` in `src/data.rs` to handle transient network failures.
- [x] Add more context to errors (e.g., which ticker failed and why).

## 3. Data Caching
**Goal**: Reduce API usage and speed up training/development.
- [x] Implement a caching layer in `src/data.rs`.
- [x] Save fetched CSV/JSON to a `.cache/` directory.
- [x] Check cache before making network requests.

## 4. Testing
**Goal**: Ensure correctness of data processing and model logic.
- [x] Add unit tests for `StockData::prepare_training_data` in `src/data.rs`.
    - Verify shape of output tensors.
    - Verify normalization (mean ~0, std ~1).
- [x] Add unit tests for `GaussianDiffusion` logic.
- [x] Add integration test with mocked network calls.

## 5. Code Quality & Documentation
**Goal**: Maintainability.
- [x] Add doc comments (`///`) to public structs and functions.
- [x] Run `cargo clippy` and fix warnings.
- [x] Refactor `train_model` to be more modular (separate data loading, model init, training loop).

## 6. Feature Enhancements
- [x] **Model Checkpointing**: Save best model based on validation loss, not just the last epoch.
- [x] **Validation Set**: Split data into Train/Validation to monitor overfitting.
- [x] **Multi-Asset Training**: Train on multiple assets simultaneously with ID embeddings.
