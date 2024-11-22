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


def calculate_historical_var_es(returns, confidence_level=0.95, rolling_window=252):
    """
    Calculate daily rolling Historical VaR and ES with proper daily updates.
    
    Parameters:
    returns: Array of returns
    confidence_level: VaR confidence level (default 0.95)
    rolling_window: Window size in days (default 252 for 1-year rolling window)
    
    Returns:
    Tuple of (VaR array, ES array) with proper daily updates
    """
    rolling_vars = []
    rolling_es = []
    
    # Cannot calculate VaR/ES until we have enough data
    for i in range(len(returns)):
        if i < rolling_window - 1:
            rolling_vars.append(np.nan)
            rolling_es.append(np.nan)
            continue
            
        # Get the window of returns ending at day i
        window = returns[i - rolling_window + 1:i + 1]
        
        # Calculate VaR
        var = np.percentile(window, (1 - confidence_level) * 100)
        rolling_vars.append(var)
        
        # Calculate ES (Expected Shortfall)
        losses = window[window <= var]
        es = np.mean(losses) if len(losses) > 0 else var
        rolling_es.append(es)
    
    return np.array(rolling_vars), np.array(rolling_es)

def calculate_lstm_var_es(model, X_test, scaler, actual_returns, confidence_level=0.95, num_scenarios=1000):
    """
    Calculate VaR and ES using LSTM predictions and Monte Carlo simulation.
    
    Parameters:
    model: Trained LSTM model
    X_test: Input sequences for prediction
    scaler: Fitted StandardScaler for inverse transformation
    actual_returns: Array of actual returns for breach calculation
    confidence_level: VaR confidence level (default 0.95)
    num_scenarios: Number of Monte Carlo scenarios (default 1000)
    
    Returns:
    Tuple of (VaR array, ES array, VaR breaches array, ES breaches array, predicted returns, predicted volatility)
    """
    device = next(model.parameters()).device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    vars = []
    es_values = []
    var_breaches = []
    es_breaches = []
    pred_means = []
    pred_vols = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            # Get predictions for single sequence
            mean, std = model(X_test_tensor[i:i+1])
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
            
            # Generate scenarios
            scenarios = np.random.normal(mean, std, (mean.shape[0], num_scenarios))
            scenarios = scaler.inverse_transform(scenarios.reshape(-1, 1)).reshape(mean.shape[0], num_scenarios)
            
            # Calculate VaR
            var = np.percentile(scenarios, (1 - confidence_level) * 100, axis=1)[0]
            vars.append(var)
            
            # Calculate ES
            losses = scenarios[0, scenarios[0] <= var]
            es = np.mean(losses) if len(losses) > 0 else var
            es_values.append(es)
            
            # Store predictions
            pred_mean = np.mean(scenarios, axis=1)[0]
            pred_vol = np.std(scenarios, axis=1)[0]
            pred_means.append(pred_mean)
            pred_vols.append(pred_vol)
            
            # Calculate breaches
            if i < len(actual_returns):
                var_breaches.append(actual_returns[i] < var)
                es_breaches.append(actual_returns[i] < es)
            else:
                var_breaches.append(False)
                es_breaches.append(False)
    
    return (np.array(vars), np.array(es_values), 
            np.array(var_breaches), np.array(es_breaches),
            np.array(pred_means), np.array(pred_vols))

def calculate_full_valuation_var_es(scenarios, confidence_level=0.95):
    """
    Calculate VaR and ES using full valuation method from scenarios.
    
    Parameters:
    scenarios: Array of shape (n_predictions, n_scenarios) containing simulated returns
    confidence_level: VaR confidence level
    
    Returns:
    Tuple of (VaR array, ES array)
    """
    vars = []
    es_values = []
    
    for scenario_set in scenarios:
        # Calculate VaR
        var = np.percentile(scenario_set, (1 - confidence_level) * 100)
        vars.append(var)
        
        # Calculate ES
        losses = scenario_set[scenario_set <= var]
        es = np.mean(losses) if len(losses) > 0 else var
        es_values.append(es)
    
    return np.array(vars), np.array(es_values)

def calculate_full_valuation_montecarlo_var_es(risk_factors, weights, confidence_level=0.95, num_scenarios=1000):
    """Standardized Monte Carlo VaR calculation"""
    lookback = min(252, len(risk_factors))
    recent_risk_factors = risk_factors[-lookback:]
    
    mean_returns = np.mean(recent_risk_factors, axis=0)
    cov_matrix = np.cov(recent_risk_factors, rowvar=False)
    
    # Generate standardized scenarios
    random_scenarios = np.random.multivariate_normal(mean_returns, cov_matrix, num_scenarios)
    portfolio_scenarios = np.dot(random_scenarios, weights)
    
    # Calculate VaR with same scale as returns
    var = np.percentile(portfolio_scenarios, (1 - confidence_level) * 100)
    es = portfolio_scenarios[portfolio_scenarios <= var].mean()
    
    return var, es, portfolio_scenarios