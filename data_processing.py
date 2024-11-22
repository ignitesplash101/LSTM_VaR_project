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


def create_sequences(data, sequence_length):
    """Create sequences for LSTM input."""
    sequences = []
    targets = []
    data = np.array(data).reshape(-1, 1)
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:(i + sequence_length)])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

def create_pca_portfolios(returns_df, market_caps_df=None, n_components=5):
    """
    Create multiple portfolios: Equal-weight S&P 500, Market-cap-weight S&P 500, and PCA portfolios
    """
    portfolios = []
    n_stocks = returns_df.shape[1]
    
    # Create equal-weight S&P 500 portfolio
    equal_weight = np.ones(n_stocks) / n_stocks
    portfolios.append(equal_weight)
    
    # Create market-cap-weight S&P 500 portfolio if market caps are available
    if market_caps_df is not None:
        market_caps = np.zeros(n_stocks)
        for i, symbol in enumerate(returns_df.columns):
            cap = market_caps_df[market_caps_df['Symbol'] == symbol]['MarketCap'].values
            if len(cap) > 0:
                market_caps[i] = cap[0]
        
        # Normalize to get weights
        market_cap_weights = market_caps / np.sum(market_caps)
        portfolios.append(market_cap_weights)
    
    # Create PCA portfolios
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_df)
    
    pca = PCA(n_components=n_components)
    pca.fit(scaled_returns)
    
    for i in range(n_components):
        weights = np.abs(pca.components_[i])
        weights = weights / np.sum(weights)
        portfolios.append(weights)
    
    return portfolios

def calculate_portfolio_returns(returns_df, weights):
    """Calculate portfolio returns."""
    returns_array = returns_df.values if isinstance(returns_df, pd.DataFrame) else returns_df
    weights_array = np.array(weights)
    return np.dot(returns_array, weights_array)

def generate_scenarios(model, X_test, num_scenarios=1000):
    """
    Generate multiple return scenarios from the LSTM predictions using Monte Carlo simulation.
    """
    device = next(model.parameters()).device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        mean, std = model(X_test_tensor)
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
    
    # Generate scenarios for each prediction point
    all_scenarios = []
    for i in range(len(mean)):
        scenarios = np.random.normal(mean[i], std[i], num_scenarios)
        all_scenarios.append(scenarios)
    
    return np.array(all_scenarios)