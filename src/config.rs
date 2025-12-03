pub const LOOKBACK: usize = 50;
pub const FORECAST: usize = 10;
pub const BATCH_SIZE: usize = 64;
pub const EPOCHS: usize = 200;
pub const LEARNING_RATE: f64 = 1e-3;

pub const TRAINING_SYMBOLS: &[&str] = &[
    "SPY", "DIA", "QQQ", "XLK", "XLI", "XLF", "XLC", "XLY", "XLRE", "XLV", "XLU", "XLP", "XLE",
    "XLB", "ARKK", "NVDA", "QQQI", "RDVI",
];
