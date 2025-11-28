mod app;
mod data;
mod diffusion;
mod inference;
mod models;
mod train;
mod tui;
mod ui;

use app::App;
use clap::Parser;
use std::io;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Train the model on historical data
    #[arg(long)]
    train: bool,

    /// Run backtest on SPY data
    #[arg(long)]
    backtest: bool,
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args = Args::parse();

    if args.train {
        match train::train_model().await {
            Ok(_) => println!("Training completed successfully."),
            Err(e) => eprintln!("Training failed: {}", e),
        }
        return Ok(());
    }

    if args.backtest {
        println!("Fetching SPY data for backtesting...");
        match data::fetch_range("SPY", "5y").await {
            Ok(data) => {
                let data = std::sync::Arc::new(data);
                match inference::run_backtest(data).await {
                    Ok(_) => println!("Backtest completed."),
                    Err(e) => eprintln!("Backtest failed: {}", e),
                }
            }
            Err(e) => eprintln!("Failed to fetch data: {}", e),
        }
        return Ok(());
    }

    let mut terminal = tui::init()?;
    let mut app = App::new();
    let res = app.run(&mut terminal).await;
    
    tui::restore()?;

    if let Err(e) = res {
        println!("Error: {:?}", e);
    }

    Ok(())
}
