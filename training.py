import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats  # Add this import
from scipy.stats import gaussian_kde
import random
from models_lstm import LSTMModel

def custom_finance_loss(pred_mean, pred_vol, targets, alpha=0.2, beta=0.3, gamma=0.2, delta=0.1, eta=1.0):
    """
    Enhanced loss function to better match return distribution characteristics and
    encourage variability in predicted volatilities.
    """
    epsilon = 1e-6
    # Ensure pred_vol is positive
    pred_vol = pred_vol.clamp(min=epsilon)  # Clamps values below epsilon to epsilon

    # Basic components
    z = (targets - pred_mean) / pred_vol

    # Negative log-likelihood with Student's t-distribution
    df = 5  # degrees of freedom
    nll = 0.5 * (torch.log(pred_vol ** 2) + (df + 1) * torch.log1p((z ** 2) / df))

    # Distribution matching components
    pred_skew = torch.mean(z ** 3)
    target_skew = -0.028  # From actual returns
    skew_loss = torch.abs(pred_skew - target_skew)

    pred_kurt = torch.mean(z ** 4)
    target_kurt = 3.578  # Normal kurtosis (3) + actual excess kurtosis (0.578)
    kurt_loss = torch.abs(pred_kurt - target_kurt)

    # Volatility matching components
    realized_vol = torch.std(targets) + epsilon
    pred_vol_mean = torch.mean(pred_vol.squeeze())

    # Mean volatility loss
    vol_mean_loss = torch.abs(pred_vol_mean - realized_vol)

    # Variance of predicted volatilities
    pred_vol_std = torch.std(pred_vol.squeeze()) + epsilon
    target_vol_std = realized_vol  # Using realized volatility as target standard deviation

    # Variance matching loss
    vol_variance_loss = torch.abs(pred_vol_std - target_vol_std)

    # Combined volatility loss
    vol_loss = gamma * vol_mean_loss + delta * vol_variance_loss

    # Combined total loss with distribution matching
    total_loss = (
        nll.mean() * 0.3 +    # Adjust weight as needed
        alpha * skew_loss +   # Skewness matching
        beta * kurt_loss +    # Kurtosis matching
        vol_loss              # Volatility matching (mean and variability)
    )

    # Calculate the breach rate in the batch
    # Using Student's t-distribution consistent with the NLL
    z_alpha = stats.t.ppf(0.05, df)  # Negative value for the 5% quantile
    z_alpha = torch.tensor(z_alpha, dtype=pred_vol.dtype, device=pred_vol.device)

    # Compute VaR
    VaR = pred_mean + z_alpha * pred_vol.squeeze()

    # Compute breaches
    breaches = (targets < VaR).float()
    batch_breach_rate = breaches.mean()
    breach_rate_loss = torch.abs(batch_breach_rate - 0.05)  # Target breach rate is 5%

    # Include in total loss with a weight
    total_loss += eta * breach_rate_loss

    return total_loss

def train_lstm_model(X_train, y_train, X_val, y_val, hidden_dim=100, num_layers=2, 
                    batch_size=32, patience=15, max_epochs=1000):
    """
    Enhanced LSTM training with improved volatility prediction and safeguards
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and optimizer
    model = LSTMModel(X_train.shape[2], hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), 
                          batch_size=batch_size, shuffle=False)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = model.state_dict()  # Initialize with current state
    counter = 0
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_vol = model(X_batch)
            loss = custom_finance_loss(pred_mean, pred_vol, y_batch)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}. Using previous best model state.")
                model.load_state_dict(best_model_state)
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
        
        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                pred_mean, pred_vol = model(X_val_batch)
                batch_loss = custom_finance_loss(pred_mean, pred_vol, y_val_batch)
                val_loss += batch_loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()  # Create a copy of the state dict
            counter = 0
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    # Ensure we have a valid model state
    if best_model_state is None:
        print("Warning: No valid model state found. Using final model state.")
        best_model_state = model.state_dict()
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    return model