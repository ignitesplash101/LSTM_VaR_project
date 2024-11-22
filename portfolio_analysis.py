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
from training import train_lstm_model
from data_processing import calculate_portfolio_returns, create_sequences, generate_scenarios
from risk_metrics import calculate_full_valuation_var_es, calculate_full_valuation_montecarlo_var_es, calculate_historical_var_es

def analyze_portfolio_var(returns_df, weights, sequence_length=252, backtest_days=378,
                        confidence_level=0.95, rolling_window=252, num_scenarios=1000):
   """Combined portfolio analysis using Historical, Monte Carlo and LSTM VaR methods."""
   # Calculate portfolio returns
   portfolio_returns = calculate_portfolio_returns(returns_df, weights)
   test_returns = portfolio_returns[-backtest_days:]
   
   # Calculate Historical VaR and ES for full period
   hist_vars, hist_es = calculate_historical_var_es(portfolio_returns, confidence_level, rolling_window)
   # Take the last backtest_days points
   hist_vars = hist_vars[-backtest_days:]
   hist_es = hist_es[-backtest_days:]
   
   # Calculate Monte Carlo VaR
   mc_vars = []
   mc_es = []
   mc_scenarios = []

   for i in range(len(test_returns)):
       end_idx = len(returns_df) - backtest_days + i
       start_idx = end_idx - rolling_window
       window_risk_factors = returns_df.iloc[start_idx:end_idx].values
       
       var, es, scenarios = calculate_full_valuation_montecarlo_var_es(
           risk_factors=window_risk_factors,
           weights=weights,
           confidence_level=confidence_level,
           num_scenarios=num_scenarios
       )
       mc_vars.append(var)
       mc_es.append(es)
       mc_scenarios.append(scenarios)
   
   # LSTM setup and training
   scaler = StandardScaler()
   scaled_returns = scaler.fit_transform(portfolio_returns.reshape(-1, 1)).flatten()
   train_data = scaled_returns[:-backtest_days]
   test_data = scaled_returns[-backtest_days:]
   
   X_train, y_train = create_sequences(train_data, sequence_length)
   X_test, y_test = create_sequences(test_data, sequence_length)
   
   val_size = min(len(X_train) // 5, 50)
   X_val, y_val = X_train[-val_size:], y_train[-val_size:]
   X_train, y_train = X_train[:-val_size], y_train[:-val_size]
   
   model = train_lstm_model(X_train, y_train, X_val, y_val)
   scenarios = generate_scenarios(model, X_test, num_scenarios)
   scenarios = scaler.inverse_transform(scenarios.reshape(-1, num_scenarios))
   
   lstm_vars, lstm_es = calculate_full_valuation_var_es(scenarios, confidence_level)
   predicted_returns = np.mean(scenarios, axis=1)
   
   # Calculate breaches and align lengths
   actual_returns = test_returns[sequence_length:]
   hist_vars_aligned = hist_vars[sequence_length:]
   hist_es_aligned = hist_es[sequence_length:]
   mc_vars = np.array(mc_vars)[sequence_length:]
   mc_es = np.array(mc_es)[sequence_length:]
   
   hist_var_breaches = actual_returns < hist_vars_aligned
   lstm_var_breaches = actual_returns < lstm_vars
   mc_var_breaches = actual_returns < mc_vars
   hist_es_breaches = actual_returns < hist_es_aligned
   lstm_es_breaches = actual_returns < lstm_es
   mc_es_breaches = actual_returns < mc_es
   
   min_len = min(len(actual_returns), len(hist_vars_aligned),
                len(hist_es_aligned), len(lstm_vars), len(lstm_es),
                len(mc_vars), len(mc_es))
   
   return (predicted_returns[:min_len], 
           hist_vars_aligned[:min_len], 
           hist_es_aligned[:min_len], 
           lstm_vars[:min_len], 
           lstm_es[:min_len],
           mc_vars[:min_len],
           mc_es[:min_len],
           hist_es_breaches[:min_len], 
           lstm_es_breaches[:min_len],
           mc_es_breaches[:min_len],
           hist_var_breaches[:min_len], 
           lstm_var_breaches[:min_len],
           mc_var_breaches[:min_len])
    
def analyze_covid_stress_period(returns_df, weights, market_event_start='2020-02-18', 
                                market_event_end='2020-03-20', rolling_window=252,
                                confidence_level=0.95, sequence_length=252, num_scenarios=1000):
    """
    Analyze portfolio behavior during COVID-19 stress period with full valuation VaR (Historical, LSTM, Monte Carlo).
    """
    returns_df = returns_df.copy()
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(calculate_portfolio_returns(returns_df, weights), 
                                  index=returns_df.index)
    
    # Define analysis period
    event_start = pd.to_datetime(market_event_start)
    event_end = pd.to_datetime(market_event_end)
    context_days = 30
    analysis_start = event_start - pd.Timedelta(days=context_days)
    analysis_end = event_end + pd.Timedelta(days=context_days)
    
    # Get ALL available training data before analysis period
    train_mask = returns_df.index < analysis_start
    train_data = portfolio_returns[train_mask]
    
    # Scale the training data
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    
    # Create sequences and train LSTM model
    X_train, y_train = create_sequences(scaled_train[:-1], sequence_length)
    
    # Split into training and validation
    val_size = min(len(X_train) // 5, 50)  # Cap validation size at 50 samples
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    # Train LSTM model
    model = train_lstm_model(
        X_train, y_train,
        X_val, y_val,
        hidden_dim=100,
        num_layers=2,
        batch_size=32,
        patience=10,
        max_epochs=200
    )
    
    # Get analysis period data
    mask = (returns_df.index >= analysis_start) & (returns_df.index <= analysis_end)
    analysis_returns = portfolio_returns[mask]
    analysis_dates = returns_df.index[mask]
    
    # Initialize results containers
    results = []
    
    # Extract risk factors (returns data for the Monte Carlo simulation)
    risk_factors = returns_df.values[-rolling_window:]
    correlation_matrix = np.corrcoef(risk_factors.T)
    
    # Process each day in analysis period
    for i, current_date in enumerate(analysis_dates):
        # Get data up to current date for Historical VaR
        hist_data = portfolio_returns[portfolio_returns.index < current_date]
        recent_window = hist_data[-rolling_window:]
        
        # Calculate Historical VaR and ES
        hist_var = np.percentile(recent_window, (1 - confidence_level) * 100)
        losses = recent_window[recent_window <= hist_var]
        hist_es = np.mean(losses) if len(losses) > 0 else hist_var
        
        # Prepare LSTM prediction sequence
        pred_data = hist_data[-sequence_length:]
        scaled_pred = scaler.transform(pred_data.values.reshape(-1, 1)).flatten()
        X_pred = scaled_pred.reshape(1, sequence_length, 1)
        X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(next(model.parameters()).device)
        
        # Generate scenarios using LSTM predictions
        with torch.no_grad():
            mean, std = model(X_pred_tensor)
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        
        # Generate scenarios for full valuation VaR using LSTM
        lstm_scenarios = np.random.normal(mean, std, (mean.shape[0], num_scenarios))
        lstm_scenarios = scaler.inverse_transform(lstm_scenarios.reshape(-1, 1)).reshape(mean.shape[0], num_scenarios)
        
        # Calculate full valuation VaR and ES from LSTM scenarios
        lstm_var = np.percentile(lstm_scenarios, (1 - confidence_level) * 100, axis=1)[0]
        lstm_losses = lstm_scenarios[0, lstm_scenarios[0] <= lstm_var]
        lstm_es = np.mean(lstm_losses) if len(lstm_losses) > 0 else lstm_var
        
        # Monte Carlo VaR and ES Calculation
        # Generate correlated scenarios based on historical risk factors
        # Fix Monte Carlo calculation in analyze_covid_stress_period function
        window_end_idx = returns_df.index.get_loc(current_date)
        window_start_idx = window_end_idx - rolling_window
        window_risk_factors = returns_df.iloc[window_start_idx:window_end_idx].values

        mc_var, mc_es, _ = calculate_full_valuation_montecarlo_var_es(
            risk_factors=window_risk_factors,
            weights=weights,
            confidence_level=confidence_level,
            num_scenarios=num_scenarios
)
        # Get actual return and check breaches
        current_return = portfolio_returns[current_date]
        
        # Store results
        results.append({
            'Date': current_date,
            'Returns': current_return,
            'Historical_VaR': hist_var,
            'Historical_ES': hist_es,
            'LSTM_VaR': lstm_var,
            'LSTM_ES': lstm_es,
            'MonteCarlo_VaR': mc_var,
            'MonteCarlo_ES': mc_es,
            'Historical_VaR_Breach': current_return < hist_var,
            'LSTM_VaR_Breach': current_return < lstm_var,
            'MonteCarlo_VaR_Breach': current_return < mc_var,
            'Is_Stress_Period': (current_date >= event_start) and (current_date <= event_end)
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(analysis_dates)} days")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate stress period statistics
    stress_period_mask = results_df['Is_Stress_Period']
    
    # Calculate statistics
    stress_stats = {
        'Stress_Period_Returns_Mean': results_df[stress_period_mask]['Returns'].mean(),
        'Stress_Period_Returns_Std': results_df[stress_period_mask]['Returns'].std(),
        'Stress_Period_Worst_Loss': results_df[stress_period_mask]['Returns'].min(),
        'Historical_VaR_Breaches': results_df[stress_period_mask]['Historical_VaR_Breach'].sum(),
        'Historical_VaR_Breach_Rate': results_df[stress_period_mask]['Historical_VaR_Breach'].mean() * 100,
        'LSTM_VaR_Breaches': results_df[stress_period_mask]['LSTM_VaR_Breach'].sum(),
        'LSTM_VaR_Breach_Rate': results_df[stress_period_mask]['LSTM_VaR_Breach'].mean() * 100,
        'MonteCarlo_VaR_Breaches': results_df[stress_period_mask]['MonteCarlo_VaR_Breach'].sum(),
        'MonteCarlo_VaR_Breach_Rate': results_df[stress_period_mask]['MonteCarlo_VaR_Breach'].mean() * 100,
        'Pre_Stress_Historical_VaR_Mean': results_df[~stress_period_mask]['Historical_VaR'].mean(),
        'Stress_Historical_VaR_Mean': results_df[stress_period_mask]['Historical_VaR'].mean(),
        'Pre_Stress_LSTM_VaR_Mean': results_df[~stress_period_mask]['LSTM_VaR'].mean(),
        'Stress_LSTM_VaR_Mean': results_df[stress_period_mask]['LSTM_VaR'].mean(),
        'Pre_Stress_MonteCarlo_VaR_Mean': results_df[~stress_period_mask]['MonteCarlo_VaR'].mean(),
        'Stress_MonteCarlo_VaR_Mean': results_df[stress_period_mask]['MonteCarlo_VaR'].mean(),
        'Historical_VaR_Change_Pct': ((results_df[stress_period_mask]['Historical_VaR'].mean() /
                                       results_df[~stress_period_mask]['Historical_VaR'].mean()) - 1) * 100,
        'LSTM_VaR_Change_Pct': ((results_df[stress_period_mask]['LSTM_VaR'].mean() /
                                 results_df[~stress_period_mask]['LSTM_VaR'].mean()) - 1) * 100,
        'MonteCarlo_VaR_Change_Pct': ((results_df[stress_period_mask]['MonteCarlo_VaR'].mean() /
                                       results_df[~stress_period_mask]['MonteCarlo_VaR'].mean()) - 1) * 100,
    }
    
    return results_df, stress_stats
