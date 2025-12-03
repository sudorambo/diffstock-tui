use crate::config::{BATCH_SIZE, EPOCHS, FORECAST, LEARNING_RATE, LOOKBACK, TRAINING_SYMBOLS};
use crate::data::{StockData, TrainingDataset};
use crate::diffusion::GaussianDiffusion;
use crate::models::time_grad::{EpsilonTheta, RNNEncoder};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap, Optimizer};
use rand::seq::SliceRandom;
use tracing::{info, error};

pub async fn train_model(
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
) -> Result<()> {
    info!("Training mode started...");
    
    info!("Configuration: Epochs={}, Batch Size={}, LR={}", 
        epochs.unwrap_or(EPOCHS), 
        batch_size.unwrap_or(BATCH_SIZE), 
        learning_rate.unwrap_or(LEARNING_RATE)
    );

    let (train_data, val_data) = fetch_training_data().await?;

    if train_data.features.is_empty() {
        return Err(anyhow::anyhow!("No training data available."));
    }

    train_model_with_data(train_data, val_data, epochs, batch_size, learning_rate).await
}

pub async fn train_model_with_data(
    train_data: TrainingDataset,
    val_data: TrainingDataset,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
) -> Result<()> {
    let device = Device::Cpu;

    let epochs = epochs.unwrap_or(EPOCHS);
    let batch_size = batch_size.unwrap_or(BATCH_SIZE);
    let learning_rate = learning_rate.unwrap_or(LEARNING_RATE);

    info!("Training Set: {} samples", train_data.features.len());
    info!("Validation Set: {} samples", val_data.features.len());

    // 2. Initialize Model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let _input_dim = 2; // Close Return, Overnight Return
    let input_dim = 2; // Close Return, Overnight Return
    let hidden_dim = 64;
    let num_layers = 3;
    let diff_steps = 100;
    let num_assets = TRAINING_SYMBOLS.len();
    
    let encoder = RNNEncoder::new(input_dim, hidden_dim, vb.pp("encoder"))?;
    let model = EpsilonTheta::new(1, hidden_dim, hidden_dim, num_layers, num_assets, vb.pp("model"))?; // input_channels=1 (target is close return)
    let diffusion = GaussianDiffusion::new(diff_steps, &device)?;

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), learning_rate)?;

    // 3. Training Loop
    let num_train_samples = train_data.features.len();
    let num_train_batches = num_train_samples / batch_size;
    
    let num_val_samples = val_data.features.len();
    let num_val_batches = if num_val_samples > 0 { num_val_samples / batch_size } else { 0 };

    let mut best_val_loss = f64::INFINITY;

    for epoch in 0..epochs {
        let mut total_train_loss = 0.0;
        
        // --- Training Phase ---
        // Shuffle indices
        let indices: Vec<usize> = (0..num_train_samples).collect();
        let mut indices = indices;
        indices.shuffle(&mut rand::thread_rng());

        for batch_idx in 0..num_train_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;
            let batch_indices = &indices[start..end];

            // Prepare Batch Tensors
            let mut batch_features = Vec::with_capacity(batch_size);
            let mut batch_targets = Vec::with_capacity(batch_size);
            let mut batch_asset_ids = Vec::with_capacity(batch_size);

            for &idx in batch_indices {
                batch_features.push(Tensor::from_slice(&train_data.features[idx], (LOOKBACK, 2), &device)?.to_dtype(DType::F32)?);
                batch_targets.push(Tensor::from_slice(&train_data.targets[idx], (FORECAST, 1), &device)?.to_dtype(DType::F32)?);
                batch_asset_ids.push(train_data.asset_ids[idx] as u32);
            }

            let x_hist = Tensor::stack(&batch_features, 0)?; 
            let x_0 = Tensor::stack(&batch_targets, 0)?;     
            let asset_ids = Tensor::new(batch_asset_ids.as_slice(), &device)?;
            
            let x_0 = x_0.permute((0, 2, 1))?; 

            // Encode History
            let cond = encoder.forward(&x_hist)?; 
            let cond = cond.unsqueeze(2)?; 

            // Sample t
            let t = Tensor::rand(0.0f32, diff_steps as f32, (batch_size,), &device)?.floor()?;
            
            let epsilon = Tensor::randn(0.0f32, 1.0f32, x_0.shape(), &device)?;
            
            let t_vec: Vec<u32> = t.to_vec1::<f32>()?.iter().map(|&x| x as u32).collect();
            let t_u32 = Tensor::new(t_vec.as_slice(), &device)?;
            
            let alpha_bar_t = diffusion.alpha_bar.index_select(&t_u32, 0)?; 
            let sqrt_alpha_bar_t = alpha_bar_t.sqrt()?;
            let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t)?.sqrt()?;
            
            let sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(1)?.unsqueeze(2)?;
            let sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(1)?.unsqueeze(2)?;
            
            let x_t = (x_0.broadcast_mul(&sqrt_alpha_bar_t)? + epsilon.broadcast_mul(&sqrt_one_minus_alpha_bar_t)?)?;
            
            let t_in = t.unsqueeze(1)?;
            let epsilon_pred = model.forward(&x_t, &t_in, &asset_ids, &cond)?;
            
            let loss = (epsilon - epsilon_pred)?.sqr()?.mean_all()?;
            
            opt.backward_step(&loss)?;
            total_train_loss += loss.to_scalar::<f32>()? as f64;
        }
        
        let avg_train_loss = total_train_loss / num_train_batches as f64;

        // --- Validation Phase ---
        let mut total_val_loss = 0.0;
        if num_val_batches > 0 {
            for batch_idx in 0..num_val_batches {
                let start = batch_idx * batch_size;
                let end = start + batch_size;
                
                // No shuffle for validation
                let mut batch_features = Vec::with_capacity(batch_size);
                let mut batch_targets = Vec::with_capacity(batch_size);
                let mut batch_asset_ids = Vec::with_capacity(batch_size);

                for idx in start..end {
                    batch_features.push(Tensor::from_slice(&val_data.features[idx], (LOOKBACK, 2), &device)?.to_dtype(DType::F32)?);
                    batch_targets.push(Tensor::from_slice(&val_data.targets[idx], (FORECAST, 1), &device)?.to_dtype(DType::F32)?);
                    batch_asset_ids.push(val_data.asset_ids[idx] as u32);
                }

                let x_hist = Tensor::stack(&batch_features, 0)?;
                let x_0 = Tensor::stack(&batch_targets, 0)?;
                let asset_ids = Tensor::new(batch_asset_ids.as_slice(), &device)?;
                let x_0 = x_0.permute((0, 2, 1))?;

                let cond = encoder.forward(&x_hist)?;
                let cond = cond.unsqueeze(2)?;

                let t = Tensor::rand(0.0f32, diff_steps as f32, (batch_size,), &device)?.floor()?;
                let epsilon = Tensor::randn(0.0f32, 1.0f32, x_0.shape(), &device)?;

                let t_vec: Vec<u32> = t.to_vec1::<f32>()?.iter().map(|&x| x as u32).collect();
                let t_u32 = Tensor::new(t_vec.as_slice(), &device)?;

                let alpha_bar_t = diffusion.alpha_bar.index_select(&t_u32, 0)?;
                let sqrt_alpha_bar_t = alpha_bar_t.sqrt()?;
                let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t)?.sqrt()?;

                let sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(1)?.unsqueeze(2)?;
                let sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(1)?.unsqueeze(2)?;

                let x_t = (x_0.broadcast_mul(&sqrt_alpha_bar_t)? + epsilon.broadcast_mul(&sqrt_one_minus_alpha_bar_t)?)?;

                let t_in = t.unsqueeze(1)?;
                let epsilon_pred = model.forward(&x_t, &t_in, &asset_ids, &cond)?;

                let loss = (epsilon - epsilon_pred)?.sqr()?.mean_all()?;
                total_val_loss += loss.to_scalar::<f32>()? as f64;
            }
        }
        
        let avg_val_loss = if num_val_batches > 0 { total_val_loss / num_val_batches as f64 } else { 0.0 };

        info!("Epoch {}: Train Loss = {:.6}, Val Loss = {:.6}", epoch + 1, avg_train_loss, avg_val_loss);

        // Checkpoint
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            info!("New best model found! Saving weights...");
            varmap.save("model_weights.safetensors")?;
        }

        if (epoch + 1) % 50 == 0 {
            let current_lr = opt.learning_rate();
            opt.set_learning_rate(current_lr * 0.5);
            info!("Decaying learning rate to {:.6}", current_lr * 0.5);
        }
    }

    info!("Training finished. Best Validation Loss: {:.6}", best_val_loss);

    Ok(())
}

async fn fetch_training_data() -> Result<(TrainingDataset, TrainingDataset)> {
    let symbols = TRAINING_SYMBOLS.to_vec();
    let mut all_features = Vec::new();
    let mut all_targets = Vec::new();
    let mut all_asset_ids = Vec::new();

    for (id, symbol) in symbols.iter().enumerate() {
        info!("Fetching data for {} (ID: {})...", symbol, id);
        match StockData::fetch_range(symbol, "5y").await {
            Ok(data) => {
                let dataset = data.prepare_training_data(LOOKBACK, FORECAST, id);
                all_features.extend(dataset.features);
                all_targets.extend(dataset.targets);
                all_asset_ids.extend(dataset.asset_ids);
            }
            Err(e) => error!("Failed to fetch {}: {}", symbol, e),
        }
    }
    
    let full_dataset = TrainingDataset {
        features: all_features,
        targets: all_targets,
        asset_ids: all_asset_ids,
    };

    // Split 80% Train, 20% Validation
    Ok(full_dataset.split(0.8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::StockData;

    #[tokio::test]
    async fn test_train_model_integration() {
        // 1. Create Mock Data
        let mock_data = StockData::new_mock("TEST", 200);
        let dataset = mock_data.prepare_training_data(LOOKBACK, FORECAST, 0);
        let (train_data, val_data) = dataset.split(0.8);

        // 2. Run Training (Short run)
        let result = train_model_with_data(
            train_data,
            val_data,
            Some(1), // 1 Epoch
            Some(16), // Small batch
            Some(1e-3)
        ).await;

        assert!(result.is_ok());
        
        // Cleanup
        if std::path::Path::new("model_weights.safetensors").exists() {
            std::fs::remove_file("model_weights.safetensors").unwrap();
        }
    }
}
