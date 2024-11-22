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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Modified architecture to encourage more spread in predictions
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Add noise during training to encourage distribution spread
        if self.training:
            noise = torch.randn_like(last_hidden) * 0.01
            last_hidden = last_hidden + noise
        
        mean = self.mean_head(last_hidden)
        vol = self.vol_head(last_hidden)
        
        return mean, vol

def create_sequences(data, sequence_length):
    """
    Create sequences of data for LSTM input with proper reshaping.
    """
    sequences = []
    targets = []
    
    if len(data) < sequence_length + 1:
        raise ValueError(f"Data length ({len(data)}) must be greater than sequence_length ({sequence_length})")
    
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # Reshape sequences to (n_sequences, sequence_length, 1)
    sequences = np.array(sequences).reshape(-1, sequence_length, 1)
    targets = np.array(targets).reshape(-1, 1)
    
    return sequences, targets

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


def calculate_rolling_var_es(returns, confidence_level=0.95, rolling_window=252):
    """
    Fixed Historical VaR calculation
    """
    rolling_vars = []
    rolling_es = []
    
    # Ensure we have enough data
    if len(returns) < rolling_window:
        raise ValueError(f"Not enough data points. Need at least {rolling_window}, got {len(returns)}")
    
    for i in range(len(returns)):
        if i < rolling_window - 1:
            rolling_vars.append(np.nan)
            rolling_es.append(np.nan)
            continue
            
        # Get the window of returns
        window = returns[max(0, i - rolling_window + 1):i + 1]
        
        # Calculate VaR
        var = np.percentile(window, (1 - confidence_level) * 100)
        
        # Calculate ES
        losses = window[window <= var]
        es = np.mean(losses) if len(losses) > 0 else var
        
        rolling_vars.append(var)
        rolling_es.append(es)
    
    return np.array(rolling_vars), np.array(rolling_es)

def calculate_portfolio_returns(returns_df, weights):
    """
    Calculate portfolio returns as weighted sum of individual stock returns.
    
    Parameters:
    returns_df: DataFrame of individual stock returns
    weights: Array of portfolio weights
    
    Returns:
    Array of portfolio returns
    """
    # Convert to numpy arrays if not already
    returns_array = returns_df.values if isinstance(returns_df, pd.DataFrame) else returns_df
    weights_array = np.array(weights)
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns_array, weights_array)
    
    return portfolio_returns

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
                    batch_size=32, patience=10, max_epochs=200):
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

def calculate_lstm_var(returns, sequence_length=252, confidence_level=0.95, num_scenarios=5000):
    """
    Calculate LSTM-based VaR with proper prediction generation
    """
    # Scale the returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns.reshape(-1, 1)).flatten()
    
    # Split into train and test ensuring full-year sequences
    train_size = len(scaled_returns) - sequence_length * 2
    train_data = scaled_returns[:train_size]
    test_data = scaled_returns[train_size:]
    
    print(f"Training data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # Train LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_lstm_model(
        X_train, y_train,
        X_test, y_test,
        hidden_dim=100,
        num_layers=2,
        batch_size=32,
        patience=10,
        max_epochs=200
    )
    
    # Generate predictions
    vars = []
    es_values = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
            mean, std = model(x)
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
            
            # Generate scenarios for this timestep
            scenarios = np.random.normal(mean[0, 0], std[0, 0], num_scenarios)
            
            # Calculate VaR and ES for this timestep
            var = np.percentile(scenarios, (1-confidence_level)*100)
            losses = scenarios[scenarios <= var]
            es = np.mean(losses) if len(losses) > 0 else var
            
            vars.append(var)
            es_values.append(es)
    
    # Convert to arrays
    vars = np.array(vars)
    es_values = np.array(es_values)
    
    # Unscale predictions
    vars = scaler.inverse_transform(vars.reshape(-1, 1)).flatten()
    es_values = scaler.inverse_transform(es_values.reshape(-1, 1)).flatten()
    
    return vars, es_values

def calculate_historical_var(returns, rolling_window=252, confidence_level=0.95):
    """
    Calculate Historical VaR using a rolling 1-year window
    """
    if len(returns) < rolling_window:
        raise ValueError(f"Need at least {rolling_window} points for Historical VaR")
    
    rolling_vars = []
    rolling_es = []
    
    for i in range(len(returns)):
        if i < rolling_window - 1:
            rolling_vars.append(np.nan)
            rolling_es.append(np.nan)
            continue
        
        # Get 1-year window
        window = returns[max(0, i - rolling_window + 1):i + 1]
        
        # Calculate VaR
        var = np.percentile(window, (1 - confidence_level) * 100)
        
        # Calculate ES
        losses = window[window <= var]
        es = np.mean(losses) if len(losses) > 0 else var
        
        rolling_vars.append(var)
        rolling_es.append(es)
    
    # Convert to arrays and remove NaN values
    rolling_vars = np.array(rolling_vars)
    rolling_es = np.array(rolling_es)
    valid_idx = ~np.isnan(rolling_vars)
    
    return rolling_vars[valid_idx], rolling_es[valid_idx]

def calculate_monte_carlo_var(returns, rolling_window=252, n_simulations=5000, confidence_level=0.95):
    """
    Calculate Monte Carlo VaR using a rolling 1-year window
    """
    if len(returns) < rolling_window:
        raise ValueError(f"Need at least {rolling_window} points for Monte Carlo VaR")
    
    rolling_vars = []
    rolling_es = []
    
    for i in range(len(returns)):
        if i < rolling_window - 1:
            rolling_vars.append(np.nan)
            rolling_es.append(np.nan)
            continue
        
        # Get 1-year window for parameter estimation
        window = returns[max(0, i - rolling_window + 1):i + 1]
        
        # Calculate parameters
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma == 0:
            sigma = np.finfo(float).eps
        
        # Generate scenarios
        scenarios = np.random.normal(mu, sigma, n_simulations)
        
        # Calculate VaR and ES
        var = np.percentile(scenarios, (1 - confidence_level) * 100)
        losses = scenarios[scenarios <= var]
        es = np.mean(losses) if len(losses) > 0 else var
        
        rolling_vars.append(var)
        rolling_es.append(es)
    
    # Convert to arrays and remove NaN values
    rolling_vars = np.array(rolling_vars)
    rolling_es = np.array(rolling_es)
    valid_idx = ~np.isnan(rolling_vars)
    
    return rolling_vars[valid_idx], rolling_es[valid_idx]

def generate_scenarios(model, X_test, num_scenarios=5000):
    """
    Generate scenarios with proper tensor handling
    """
    device = next(model.parameters()).device
    scenarios = []
    
    with torch.no_grad():
        for x in X_test:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            mean, std = model(x)
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
            scenario = np.random.normal(mean, std, (1, num_scenarios))
            scenarios.append(scenario[0])
    
    return np.array(scenarios)

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

def analyze_portfolio_var(returns_df, weights, sequence_length=252, backtest_days=378,
                        confidence_level=0.95, rolling_window=252, num_scenarios=5000):
    """
    Combined analysis using separated VaR calculations with proper array alignment
    """
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(returns_df, weights)
    
    # Ensure enough data for all calculations
    min_required = max(sequence_length * 2, rolling_window) + backtest_days
    if len(portfolio_returns) < min_required:
        raise ValueError(f"Need at least {min_required} data points, got {len(portfolio_returns)}")
    
    # Get required data window
    analysis_returns = portfolio_returns[-min_required:]
    
    # Calculate Historical VaR for the backtest period
    print("Calculating Historical VaR...")
    backtest_returns = analysis_returns[-backtest_days:]
    hist_vars, hist_es = calculate_historical_var(
        backtest_returns,
        rolling_window=rolling_window,
        confidence_level=confidence_level
    )
    
    print("Calculating Monte Carlo VaR...")
    mc_vars, mc_es = calculate_monte_carlo_var(
        backtest_returns,
        rolling_window=rolling_window,
        n_simulations=num_scenarios,
        confidence_level=confidence_level
    )
    
    print("Calculating LSTM VaR...")
    # Scale the returns for LSTM
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(analysis_returns.reshape(-1, 1)).flatten()
    
    # Create sequences for LSTM
    X_train, y_train = create_sequences(scaled_returns[:-backtest_days], sequence_length)
    X_test, y_test = create_sequences(scaled_returns[-backtest_days-sequence_length:], sequence_length)
    
    # Train LSTM model
    model = train_lstm_model(
        X_train, y_train,
        X_test, y_test,
        hidden_dim=100,
        num_layers=2,
        batch_size=32,
        patience=10,
        max_epochs=200
    )
    
    # Generate predictions and scenarios
    lstm_predictions = []
    lstm_vars = []
    lstm_es = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for x in X_test:
            x = torch.tensor(x.reshape(1, sequence_length, 1), dtype=torch.float32).to(device)
            mean, std = model(x)
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
            
            # Store the mean prediction
            pred = mean[0, 0]
            lstm_predictions.append(pred)
            
            # Generate scenarios for VaR calculation
            scenarios = np.random.normal(mean[0, 0], std[0, 0], num_scenarios)
            
            # Calculate VaR and ES
            var = np.percentile(scenarios, (1-confidence_level)*100)
            losses = scenarios[scenarios <= var]
            es = np.mean(losses) if len(losses) > 0 else var
            
            lstm_vars.append(var)
            lstm_es.append(es)
    
    # Convert predictions and risk measures to arrays
    lstm_predictions = np.array(lstm_predictions)
    lstm_vars = np.array(lstm_vars)
    lstm_es = np.array(lstm_es)
    
    # Unscale predictions and risk measures
    lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
    lstm_vars = scaler.inverse_transform(lstm_vars.reshape(-1, 1)).flatten()
    lstm_es = scaler.inverse_transform(lstm_es.reshape(-1, 1)).flatten()
    
    # Get test returns aligned with predictions
    test_returns = backtest_returns[:len(lstm_predictions)]
    
    # Align all arrays to the same length
    min_len = min(len(test_returns), len(lstm_predictions), len(hist_vars), 
                 len(hist_es), len(lstm_vars), len(lstm_es), 
                 len(mc_vars), len(mc_es))
    
    test_returns = test_returns[:min_len]
    lstm_predictions = lstm_predictions[:min_len]
    hist_vars = hist_vars[:min_len]
    hist_es = hist_es[:min_len]
    lstm_vars = lstm_vars[:min_len]
    lstm_es = lstm_es[:min_len]
    mc_vars = mc_vars[:min_len]
    mc_es = mc_es[:min_len]
    
    # Calculate breaches using aligned arrays
    hist_es_breaches = test_returns < hist_es
    lstm_es_breaches = test_returns < lstm_es
    mc_es_breaches = test_returns < mc_es
    
    print(f"\nArray lengths after alignment:")
    print(f"Test returns: {len(test_returns)}")
    print(f"LSTM predictions: {len(lstm_predictions)}")
    print(f"Historical VaR: {len(hist_vars)}")
    print(f"LSTM VaR: {len(lstm_vars)}")
    print(f"Monte Carlo VaR: {len(mc_vars)}")
    
    return (lstm_predictions, hist_vars, hist_es, lstm_vars, lstm_es, mc_vars, mc_es,
            hist_es_breaches, lstm_es_breaches, mc_es_breaches)

def plot_risk_measures_comparison(portfolio_id, dates, returns, hist_vars, hist_es, 
                                lstm_vars, lstm_es, mc_vars, mc_es,
                                hist_es_breaches, lstm_es_breaches, mc_es_breaches,
                                plots_dir):
    """
    Fixed plotting function with explicit line styles and zorder to ensure all VaR measures are visible
    """
    # Calculate VaR breaches
    hist_var_breaches = returns < hist_vars
    lstm_var_breaches = returns < lstm_vars
    mc_var_breaches = returns < mc_vars
    
    # Calculate breach rates
    hist_var_breach_rate = np.mean(hist_var_breaches) * 100
    lstm_var_breach_rate = np.mean(lstm_var_breaches) * 100
    mc_var_breach_rate = np.mean(mc_var_breaches) * 100
    hist_es_breach_rate = np.mean(hist_es_breaches) * 100
    lstm_es_breach_rate = np.mean(lstm_es_breaches) * 100
    mc_es_breach_rate = np.mean(mc_es_breaches) * 100

    fig, axs = plt.subplots(2, 1, figsize=(16, 16), constrained_layout=True)

    # Top Plot: Returns and VaR
    ax_var = axs[0]
    # Plot returns first (background)
    ax_var.plot(dates, returns, label="Actual Returns", color="blue", alpha=0.7, zorder=1)
    
    # Plot VaR lines with different line styles and higher zorder
    ax_var.plot(dates, hist_vars, label="Historical VaR", color="red", 
                linestyle='--', linewidth=1.5, zorder=2)
    ax_var.plot(dates, lstm_vars, label="LSTM VaR", color="green", 
                linestyle='--', linewidth=1.5, zorder=2)
    ax_var.plot(dates, mc_vars, label="Monte Carlo VaR", color="purple", 
                linestyle=':', linewidth=2, zorder=2)
    
    # Plot breaches with highest zorder
    ax_var.scatter(dates[hist_var_breaches], returns[hist_var_breaches], 
                  color="red", marker="x", label="Historical VaR Breach", zorder=5)
    ax_var.scatter(dates[lstm_var_breaches], returns[lstm_var_breaches], 
                  color="green", marker="x", label="LSTM VaR Breach", zorder=5)
    ax_var.scatter(dates[mc_var_breaches], returns[mc_var_breaches], 
                  color="purple", marker="x", label="MC VaR Breach", zorder=5)

    breach_text = (
        f"Historical VaR Breach Rate: {hist_var_breach_rate:.2f}%\n"
        f"LSTM VaR Breach Rate: {lstm_var_breach_rate:.2f}%\n"
        f"Monte Carlo VaR Breach Rate: {mc_var_breach_rate:.2f}%"
    )
    ax_var.text(0.02, 0.98, breach_text, transform=ax_var.transAxes,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax_var.set_title(f"{portfolio_id}: Returns and VaR Measures")
    ax_var.set_xlabel("Date")
    ax_var.set_ylabel("Returns")
    ax_var.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax_var.grid(True, alpha=0.3)

    # Bottom Plot: Returns and ES
    ax_es = axs[1]
    # Plot returns first (background)
    ax_es.plot(dates, returns, label="Actual Returns", color="blue", alpha=0.7, zorder=1)
    
    # Plot ES lines with different line styles and higher zorder
    ax_es.plot(dates, hist_es, label="Historical ES", color="red", 
               linestyle='--', linewidth=1.5, zorder=2)
    ax_es.plot(dates, lstm_es, label="LSTM ES", color="green", 
               linestyle='--', linewidth=1.5, zorder=2)
    ax_es.plot(dates, mc_es, label="Monte Carlo ES", color="purple", 
               linestyle=':', linewidth=2, zorder=2)
    
    # Plot ES breaches with highest zorder
    ax_es.scatter(dates[hist_es_breaches], returns[hist_es_breaches], 
                 color="red", marker="x", label="Historical ES Breach", zorder=5)
    ax_es.scatter(dates[lstm_es_breaches], returns[lstm_es_breaches], 
                 color="green", marker="x", label="LSTM ES Breach", zorder=5)
    ax_es.scatter(dates[mc_es_breaches], returns[mc_es_breaches], 
                 color="purple", marker="x", label="MC ES Breach", zorder=5)

    breach_text = (
        f"Historical ES Breach Rate: {hist_es_breach_rate:.2f}%\n"
        f"LSTM ES Breach Rate: {lstm_es_breach_rate:.2f}%\n"
        f"Monte Carlo ES Breach Rate: {mc_es_breach_rate:.2f}%"
    )
    ax_es.text(0.02, 0.98, breach_text, transform=ax_es.transAxes,
               verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax_es.set_title(f"{portfolio_id}: Expected Shortfall Measures")
    ax_es.set_xlabel("Date")
    ax_es.set_ylabel("Returns")
    ax_es.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax_es.grid(True, alpha=0.3)

    # Ensure all data is plotted with proper limits
    for ax in axs:
        ax.set_xlim(dates[0], dates[-1])
        min_val = min(returns.min(), hist_vars.min(), lstm_vars.min(), mc_vars.min()) * 1.1
        max_val = max(returns.max(), hist_vars.max(), lstm_vars.max(), mc_vars.max()) * 1.1
        ax.set_ylim(min_val, max_val)

    save_path = os.path.join(plots_dir, f"{portfolio_id}_risk_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path

def plot_portfolio_risk_metrics(portfolio_id, dates, returns, rolling_vars, 
                              rolling_es, plots_dir):
    """
    Modified plotting function with improved visualization.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # Plot returns and rolling VaR
    ax_perf = axs[0]
    ax_perf.plot(dates, returns, label="Actual Returns", color="blue", alpha=0.7)
    ax_perf.axhline(0, linestyle="--", color="gray", label="Zero Line")
    ax_perf.plot(dates, rolling_vars, linestyle="--", color="red", label="Rolling VaR (95%)")
    
    # Highlight VaR breaches
    breaches = returns < rolling_vars
    ax_perf.fill_between(dates, returns, rolling_vars,
                        where=breaches, color="red", alpha=0.3, label="VaR Breach")
    
    breach_rate = np.mean(breaches) * 100
    ax_perf.text(0.02, 0.98, 
                f'VaR Breach Rate: {breach_rate:.2f}%\n'
                f'Total Breaches: {np.sum(breaches)}',
                transform=ax_perf.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_perf.set_title(f"Portfolio {portfolio_id}: Actual Returns and Rolling VaR")
    ax_perf.set_xlabel("Date")
    ax_perf.set_ylabel("Returns")
    ax_perf.legend()
    ax_perf.grid(True)
    
    # Plot risk measures
    ax_risk = axs[1]
    ax_risk.plot(dates, rolling_vars, label="Rolling VaR", color="red")
    ax_risk.plot(dates, rolling_es, label="Rolling ES", color="blue")
    ax_risk.axhline(0, linestyle="--", color="gray", label="Zero Line")
    ax_risk.set_title(f"Portfolio {portfolio_id}: Risk Measures Over Time")
    ax_risk.set_xlabel("Date")
    ax_risk.set_ylabel("Risk Measure")
    ax_risk.legend()
    ax_risk.grid(True)
    
    save_path = os.path.join(plots_dir, f"portfolio_{portfolio_id}_risk_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path, breach_rate

def plot_return_distributions(portfolio_id, predicted_returns, actual_returns, plots_dir):
    """
    Modified distribution plots to correctly show differences between predicted and actual returns
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Calculate statistics for predicted returns
    mu_pred = np.mean(predicted_returns)
    sigma_pred = np.std(predicted_returns)
    skew_pred = stats.skew(predicted_returns)
    kurt_pred = stats.kurtosis(predicted_returns)
    
    # 1. Histogram of predicted returns with normal fit
    ax1 = fig.add_subplot(gs[0, 0])
    n_pred, bins_pred, _ = ax1.hist(predicted_returns, bins=50, density=True, 
                                   alpha=0.7, color='blue', label='Predicted Returns')
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu_pred, sigma_pred)
    ax1.plot(x, p, 'k--', linewidth=2, label='Normal Fit')
    ax1.set_title(f'{portfolio_id}: Distribution of Predicted Returns')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Stats box for predicted returns
    stats_text = (f'Mean: {mu_pred:.6f}\n'
                 f'Std: {sigma_pred:.6f}\n'
                 f'Skewness: {skew_pred:.3f}\n'
                 f'Kurtosis: {kurt_pred:.3f}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Q-Q plot for predicted returns
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(predicted_returns, dist="norm", plot=ax2)
    ax2.set_title(f'{portfolio_id}: Q-Q Plot of Predicted Returns')
    
    # Calculate statistics for actual returns
    mu_act = np.mean(actual_returns)
    sigma_act = np.std(actual_returns)
    skew_act = stats.skew(actual_returns)
    kurt_act = stats.kurtosis(actual_returns)
    
    # 3. Histogram of actual returns with normal fit
    ax3 = fig.add_subplot(gs[1, 0])
    n_act, bins_act, _ = ax3.hist(actual_returns, bins=50, density=True, 
                                 alpha=0.7, color='green', label='Actual Returns')
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu_act, sigma_act)
    ax3.plot(x, p, 'k--', linewidth=2, label='Normal Fit')
    ax3.set_title(f'{portfolio_id}: Distribution of Actual Returns')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Density')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Stats box for actual returns
    stats_text = (f'Mean: {mu_act:.6f}\n'
                 f'Std: {sigma_act:.6f}\n'
                 f'Skewness: {skew_act:.3f}\n'
                 f'Kurtosis: {kurt_act:.3f}')
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Q-Q plot for actual returns
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(actual_returns, dist="norm", plot=ax4)
    ax4.set_title(f'{portfolio_id}: Q-Q Plot of Actual Returns')
    
    # 5. Distribution comparison
    ax5 = fig.add_subplot(gs[:, 2])
    
    # Create common bins for both distributions
    all_returns = np.concatenate([predicted_returns, actual_returns])
    min_val = np.min(all_returns)
    max_val = np.max(all_returns)
    bins = np.linspace(min_val, max_val, 50)
    
    # Plot both histograms with transparency
    ax5.hist(predicted_returns, bins=bins, density=True, alpha=0.5,
             color='blue', label='Predicted Returns')
    ax5.hist(actual_returns, bins=bins, density=True, alpha=0.5,
             color='green', label='Actual Returns')
    
    # Add KDE curves
    kde_pred = gaussian_kde(predicted_returns)
    kde_act = gaussian_kde(actual_returns)
    x_range = np.linspace(min_val, max_val, 200)
    ax5.plot(x_range, kde_pred(x_range), color='blue', linestyle='-', alpha=0.8)
    ax5.plot(x_range, kde_act(x_range), color='green', linestyle='-', alpha=0.8)
    
    ax5.set_title(f'{portfolio_id}: Return Distributions Comparison')
    ax5.set_xlabel('Returns')
    ax5.set_ylabel('Density')
    ax5.legend(loc='upper right')
    ax5.grid(True)
    
    # Add statistical comparison
    _, p_value_pred = stats.normaltest(predicted_returns)
    _, p_value_act = stats.normaltest(actual_returns)
    # Calculate KS test between distributions
    ks_stat, ks_pvalue = stats.ks_2samp(predicted_returns, actual_returns)
    
    test_text = (f'Normality Test p-values:\n'
                f'Predicted: {p_value_pred:.6f}\n'
                f'Actual: {p_value_act:.6f}\n\n'
                f'Distribution Comparison:\n'
                f'KS-test statistic: {ks_stat:.6f}\n'
                f'KS-test p-value: {ks_pvalue:.6f}\n\n'
                f'Standard Deviations:\n'
                f'Predicted: {sigma_pred:.6f}\n'
                f'Actual: {sigma_act:.6f}')
    ax5.text(0.02, 0.98, test_text, transform=ax5.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"portfolio_{portfolio_id}_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_covid_stress_period(returns_df, weights, market_event_start='2020-02-18', 
                              market_event_end='2020-03-20', rolling_window=252,
                              confidence_level=0.95, sequence_length=252, num_scenarios=5000):
    """
    Analyze portfolio behavior during COVID-19 stress period with full valuation VaR
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
    
    print("\nTraining Data Summary:")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Number of training days: {len(train_data)}")
    print(f"Number of years of training data: {len(train_data)/252:.2f} years")
    
    # Scale the training data
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    
    print("\nCreating sequences and training LSTM...")
    X_train, y_train = create_sequences(scaled_train[:-1], sequence_length)
    
    # Split into training and validation
    val_size = min(len(X_train) // 5, 50)  # Cap validation size at 50 samples
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    
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
    print("LSTM training completed.")
    
    # Get analysis period data
    mask = (returns_df.index >= analysis_start) & (returns_df.index <= analysis_end)
    analysis_returns = portfolio_returns[mask]
    analysis_dates = returns_df.index[mask]
    
    print("\nAnalysis Period Summary:")
    print(f"Analysis period: {analysis_dates[0]} to {analysis_dates[-1]}")
    print(f"Number of analysis days: {len(analysis_dates)}")
    
    # Initialize results containers
    results = []
    
    # Process each day in analysis period
    print("Processing analysis period...")
    for i, current_date in enumerate(analysis_dates):
        # Get data up to current date for Historical VaR
        hist_data = portfolio_returns[portfolio_returns.index < current_date]
        recent_window = hist_data[-rolling_window:]
        
        # Calculate Historical VaR and ES
        hist_var = np.percentile(recent_window, (1 - confidence_level) * 100)
        losses = recent_window[recent_window <= hist_var]
        hist_es = np.mean(losses) if len(losses) > 0 else hist_var
        
        # Calculate Monte Carlo VaR
        mu = np.mean(recent_window)
        sigma = np.std(recent_window)
        mc_scenarios = np.random.normal(mu, sigma, num_scenarios)
        mc_var = np.percentile(mc_scenarios, (1 - confidence_level) * 100)
        mc_losses = mc_scenarios[mc_scenarios <= mc_var]
        mc_es = np.mean(mc_losses) if len(mc_losses) > 0 else mc_var
        
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
            
        # Generate scenarios for full valuation VaR
        scenarios = np.random.normal(mean, std, (mean.shape[0], num_scenarios))
        scenarios = scaler.inverse_transform(scenarios.reshape(-1, 1)).reshape(mean.shape[0], num_scenarios)
        
        # Calculate full valuation VaR and ES from scenarios
        lstm_var = np.percentile(scenarios, (1 - confidence_level) * 100, axis=1)[0]
        losses = scenarios[0, scenarios[0] <= lstm_var]
        lstm_es = np.mean(losses) if len(losses) > 0 else lstm_var
        
        # Get predicted return (mean of scenarios)
        pred_mean = np.mean(scenarios, axis=1)[0]
        pred_vol = np.std(scenarios, axis=1)[0]
        
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
            'MC_VaR': mc_var,
            'MC_ES': mc_es,
            'LSTM_Predicted_Returns': pred_mean,
            'LSTM_Predicted_Volatility': pred_vol,
            'Historical_VaR_Breach': current_return < hist_var,
            'LSTM_VaR_Breach': current_return < lstm_var,
            'MC_VaR_Breach': current_return < mc_var,
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
        'MC_VaR_Breaches': results_df[stress_period_mask]['MC_VaR_Breach'].sum(),
        'MC_VaR_Breach_Rate': results_df[stress_period_mask]['MC_VaR_Breach'].mean() * 100,
        'Pre_Stress_Historical_VaR_Mean': results_df[~stress_period_mask]['Historical_VaR'].mean(),
        'Stress_Historical_VaR_Mean': results_df[stress_period_mask]['Historical_VaR'].mean(),
        'Pre_Stress_LSTM_VaR_Mean': results_df[~stress_period_mask]['LSTM_VaR'].mean(),
        'Stress_LSTM_VaR_Mean': results_df[stress_period_mask]['LSTM_VaR'].mean(),
        'Pre_Stress_MC_VaR_Mean': results_df[~stress_period_mask]['MC_VaR'].mean(),
        'Stress_MC_VaR_Mean': results_df[stress_period_mask]['MC_VaR'].mean(),
        'Historical_VaR_Change_Pct': ((results_df[stress_period_mask]['Historical_VaR'].mean() /
                                     results_df[~stress_period_mask]['Historical_VaR'].mean()) - 1) * 100,
        'LSTM_VaR_Change_Pct': ((results_df[stress_period_mask]['LSTM_VaR'].mean() /
                                results_df[~stress_period_mask]['LSTM_VaR'].mean()) - 1) * 100,
        'MC_VaR_Change_Pct': ((results_df[stress_period_mask]['MC_VaR'].mean() /
                              results_df[~stress_period_mask]['MC_VaR'].mean()) - 1) * 100,
        'LSTM_Prediction_MSE': np.mean((results_df['Returns'] - results_df['LSTM_Predicted_Returns'])**2),
        'LSTM_Prediction_MAE': np.mean(np.abs(results_df['Returns'] - results_df['LSTM_Predicted_Returns'])),
        'LSTM_Prediction_RMSE': np.sqrt(np.mean((results_df['Returns'] - results_df['LSTM_Predicted_Returns'])**2)),
        'Training_Start_Date': str(train_data.index[0]),
        'Training_End_Date': str(train_data.index[-1]),
        'Training_Days': len(train_data),
        'Training_Years': len(train_data)/252
    }
    
    return results_df, stress_stats

def plot_stress_period_analysis_covid(results_df, stress_stats, portfolio_name, plots_dir):
    """
    Enhanced stress period analysis plot including Monte Carlo VaR.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)

    # 1. Returns and Risk Measures Plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results_df['Date'], results_df['Returns'], label='Portfolio Returns', color='blue', alpha=0.7)
    ax1.plot(results_df['Date'], results_df['Historical_VaR'], label='Historical VaR', color='red', linestyle='--')
    ax1.plot(results_df['Date'], results_df['LSTM_VaR'], label='LSTM VaR', color='green', linestyle='--')
    ax1.plot(results_df['Date'], results_df['MC_VaR'], label='Monte Carlo VaR', color='purple', linestyle='--')
    
    # Highlight stress period
    stress_mask = results_df['Is_Stress_Period']
    stress_dates = results_df[stress_mask]['Date']
    ax1.axvspan(stress_dates.iloc[0], stress_dates.iloc[-1],
                color='gray', alpha=0.2, label='COVID-19 Stress Period')

    # Plot breaches for each method
    for method, color in [('Historical_VaR_Breach', 'red'), 
                         ('LSTM_VaR_Breach', 'green'),
                         ('MC_VaR_Breach', 'purple')]:
        breach_dates = results_df[results_df[method]]['Date']
        breach_returns = results_df[results_df[method]]['Returns']
        ax1.scatter(breach_dates, breach_returns, color=color, marker='x',
                   label=f'{method.split("_")[0]} Breaches', zorder=5)

    ax1.set_title(f'{portfolio_name}: Returns and VaR Measures During COVID-19 Stress Period')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Return/Risk Level')
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
    ax1.grid(True)

    # Add stress period statistics
    stats_text = (
        f"Stress Period Statistics:\n"
        f"Mean Return: {stress_stats['Stress_Period_Returns_Mean']:.4%}\n"
        f"Worst Loss: {stress_stats['Stress_Period_Worst_Loss']:.4%}\n"
        f"Historical VaR Breach Rate: {stress_stats['Historical_VaR_Breach_Rate']:.1f}%\n"
        f"LSTM VaR Breach Rate: {stress_stats['LSTM_VaR_Breach_Rate']:.1f}%\n"
        f"MC VaR Breach Rate: {stress_stats['MC_VaR_Breach_Rate']:.1f}%"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. VaR Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results_df['Date'], results_df['Historical_VaR'], label='Historical VaR', color='red')
    ax2.plot(results_df['Date'], results_df['LSTM_VaR'], label='LSTM VaR', color='green')
    ax2.plot(results_df['Date'], results_df['MC_VaR'], label='Monte Carlo VaR', color='purple')
    ax2.axvspan(stress_dates.iloc[0], stress_dates.iloc[-1],
                color='gray', alpha=0.2, label='COVID-19 Stress Period')
    ax2.set_title('VaR Methods Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('VaR')
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True)

    # 3. Breach Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    # Calculate cumulative breaches for each method
    for method, color, label in [('Historical_VaR_Breach', 'red', 'Historical'), 
                                ('LSTM_VaR_Breach', 'green', 'LSTM'),
                                ('MC_VaR_Breach', 'purple', 'Monte Carlo')]:
        cum_breaches = np.cumsum(results_df[method])
        ax3.plot(results_df['Date'], cum_breaches, 
                label=f'Cumulative {label} Breaches', color=color)

    ax3.axvspan(stress_dates.iloc[0], stress_dates.iloc[-1],
                color='gray', alpha=0.2, label='COVID-19 Stress Period')
    ax3.set_title('Cumulative VaR Breaches')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Number of Breaches')
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax3.grid(True)

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(plots_dir, f"{portfolio_name}_covid_stress_analysis_with_mc.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path

def main():
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Set up directories - Modified path handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", 'data')  # Removed nested "data" directory
    results_dir = os.path.join(current_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    covid_analysis_dir = os.path.join(results_dir, "covid_analysis")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(covid_analysis_dir, exist_ok=True)
    
    # Set parameters
    backtest_days = 378  # ~1.5 years
    rolling_window = 252  # 1 year
    sequence_length = 252  # 1 year for LSTM
    confidence_level = 0.95
    num_scenarios = 5000
    
    # Load data - Updated file paths
    data_path = os.path.join(data_dir, "sp500_adjusted_close_cleaned.csv")
    market_caps_path = os.path.join(data_dir, "sp500_market_caps.csv")
    
    # Check if required files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Required file not found: {data_path}")
    if not os.path.exists(market_caps_path):
        raise FileNotFoundError(f"Required file not found: {market_caps_path}")
    
    print("Loading data...")
    data = pd.read_csv(data_path)
    market_caps = pd.read_csv(market_caps_path)
    
    data['Date'] = pd.to_datetime(data['Date'])
    dates = data['Date'].values
    
    # Calculate returns
    log_prices = np.log(data.iloc[:, 1:])
    returns_df = log_prices.diff().dropna()
    returns_df.index = pd.to_datetime(data['Date'].values[1:])  # Set proper datetime index
    dates = returns_df.index.values  # Aligned dates
    
    print("Creating portfolios...")
    # Create portfolios: Equal-weight S&P 500, Market-cap-weight S&P 500, and PCA portfolios
    portfolios = create_pca_portfolios(returns_df, market_caps, n_components=5)
    
    # Analyze each portfolio
    for i, weights in enumerate(portfolios):
        # Determine portfolio name
        if i == 0:
            portfolio_name = "S&P 500 (Equal-Weight)"
        elif i == 1:
            portfolio_name = "S&P 500 (Market-Cap-Weight)"
        else:
            portfolio_name = f"PCA Portfolio {i-1}"
            
        print(f"\nAnalyzing {portfolio_name}...")
        
        # Calculate portfolio returns
        portfolio_returns = calculate_portfolio_returns(returns_df, weights)
        
        # Calculate Monte Carlo VaR for the backtest period
        mc_vars, mc_es = calculate_monte_carlo_var(
            portfolio_returns[-backtest_days:],
            rolling_window=rolling_window,
            n_simulations=5000,
            confidence_level=confidence_level
        )
        
        # Get LSTM and Historical VaR metrics
        predicted_returns, hist_vars, hist_es, lstm_vars, lstm_es, mc_vars, mc_es, \
        hist_es_breaches, lstm_es_breaches, mc_es_breaches = analyze_portfolio_var(
            returns_df, weights, 
            backtest_days=backtest_days,
            confidence_level=confidence_level, 
            rolling_window=rolling_window,
            num_scenarios=5000
        )
        
        # Prepare data for plotting
        plot_dates = dates[-backtest_days:]
        plot_returns = portfolio_returns[-backtest_days:]
        
        # Align all series lengths
        min_len = min(len(plot_dates), len(plot_returns), len(hist_vars), 
                    len(hist_es), len(predicted_returns), len(mc_vars))
        
        plot_dates = plot_dates[-min_len:]
        plot_returns = plot_returns[-min_len:]
        hist_vars = hist_vars[-min_len:]
        hist_es = hist_es[-min_len:]
        predicted_returns = predicted_returns[-min_len:]
        lstm_vars = lstm_vars[-min_len:]
        lstm_es = lstm_es[-min_len:]
        mc_vars = mc_vars[-min_len:]
        mc_es = mc_es[-min_len:]
        
        # Calculate breach rates
        hist_var_breaches = plot_returns < hist_vars
        lstm_var_breaches = plot_returns < lstm_vars
        mc_var_breaches = plot_returns < mc_vars
        mc_es_breaches = plot_returns < mc_es
        
        mse_predicted_returns = np.mean((predicted_returns - plot_returns) ** 2)

        # Update portfolio statistics
        portfolio_stats = pd.Series({
            'Annual_Return': np.mean(portfolio_returns) * 252,
            'Annual_Volatility': np.std(portfolio_returns) * np.sqrt(252),
            'Sharpe_Ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252),
            'Historical_VaR_Breach_Rate': hist_var_breaches.mean() * 100,
            'LSTM_VaR_Breach_Rate': lstm_var_breaches.mean() * 100,
            'MC_VaR_Breach_Rate': mc_var_breaches.mean() * 100,
            'Skewness': stats.skew(portfolio_returns),
            'Excess_Kurtosis': stats.kurtosis(portfolio_returns),
            'Max_Drawdown': np.min(portfolio_returns),
            'VaR_95_Historical': np.percentile(portfolio_returns, 5),
            'VaR_95_MC': mc_vars.mean()
        })
        
        # Create weights DataFrame
        weights_df = pd.DataFrame({
            'Stock': returns_df.columns,
            'Weight': weights,
            'Risk_Contribution': weights * np.std(returns_df, axis=0),
            'Expected_Shortfall_Contribution': weights * np.mean(returns_df[returns_df < np.percentile(returns_df, 5)], axis=0)
        })
        weights_df['Weight_Percentage'] = weights_df['Weight'] * 100
        weights_df = weights_df.sort_values('Weight_Percentage', ascending=False)
                
        # Update summary statistics
        summary_stats = pd.Series({
            'Historical_VaR_Breach_Rate': hist_var_breaches.mean() * 100,
            'LSTM_VaR_Breach_Rate': lstm_var_breaches.mean() * 100,
            'MC_VaR_Breach_Rate': mc_var_breaches.mean() * 100,
            'Avg_Historical_VaR': hist_vars.mean(),
            'Avg_Historical_ES': hist_es.mean(),
            'Avg_LSTM_VaR': lstm_vars.mean(),
            'Avg_LSTM_ES': lstm_es.mean(),
            'Avg_MC_VaR': mc_vars.mean(),
            'Avg_MC_ES': mc_es.mean(),
            'LSTM_Prediction_MAE': np.mean(np.abs(plot_returns - predicted_returns)),
            'LSTM_Prediction_MSE': mse_predicted_returns,
            'LSTM_Prediction_Std': np.std(predicted_returns),
            'Actual_Returns_Std': np.std(plot_returns),
            'LSTM_VaR_Coverage_Ratio': (lstm_var_breaches.mean() * 100) / 5.0,
            'Historical_VaR_Coverage_Ratio': (hist_var_breaches.mean() * 100) / 5.0,
            'MC_VaR_Coverage_Ratio': (mc_var_breaches.mean() * 100) / 5.0
        })
        
        # Update risk metrics DataFrame
        risk_metrics_df = pd.DataFrame({
            'Date': plot_dates,
            'Portfolio': portfolio_name,
            'Actual_Returns': plot_returns,
            'LSTM_Predicted_Returns': predicted_returns,
            'Historical_VaR': hist_vars,
            'Historical_ES': hist_es,
            'LSTM_VaR': lstm_vars,
            'LSTM_ES': lstm_es,
            'MC_VaR': mc_vars,
            'MC_ES': mc_es,
            'Historical_VaR_Breach': hist_var_breaches.astype(int),
            'LSTM_VaR_Breach': lstm_var_breaches.astype(int),
            'MC_VaR_Breach': mc_var_breaches.astype(int),
            'Historical_ES_Breach': hist_es_breaches.astype(int),
            'LSTM_ES_Breach': lstm_es_breaches.astype(int),
            'MC_ES_Breach': mc_es_breaches.astype(int)
        })
        
        # Update breach rates summary
        breach_rates = pd.DataFrame({
            'Date': ['Breach Rates'],
            'Portfolio': [portfolio_name],
            'Historical_VaR_Breach': [np.mean(hist_var_breaches) * 100],
            'LSTM_VaR_Breach': [np.mean(lstm_var_breaches) * 100],
            'MC_VaR_Breach': [np.mean(mc_var_breaches) * 100],
            'Historical_ES_Breach': [np.mean(hist_es_breaches) * 100],
            'LSTM_ES_Breach': [np.mean(lstm_es_breaches) * 100],
            'MC_ES_Breach': [np.mean(mc_es_breaches) * 100]
        })

        risk_metrics_df = pd.concat([risk_metrics_df, breach_rates], ignore_index=True)

        # Create plots with Monte Carlo VaR
        risk_plot_path = plot_risk_measures_comparison(
                portfolio_name,
                plot_dates,
                plot_returns,
                hist_vars,
                hist_es,
                lstm_vars,
                lstm_es,
                mc_vars,
                mc_es,
                hist_es_breaches,
                lstm_es_breaches,
                mc_es_breaches,
                plots_dir
            )
        print(f"Saved risk measures comparison plot to: {risk_plot_path}")
    # Add COVID-19 Stress Period Analysis
    
        print(f"\nAnalyzing COVID-19 stress period for {portfolio_name}...")
        market_event_start = '2020-02-18'
        market_event_end = '2020-03-20'
        
        # Run COVID stress period analysis with LSTM comparison
        covid_results_df, covid_stress_stats = analyze_covid_stress_period(
            returns_df,
            weights,
            market_event_start=market_event_start,
            market_event_end=market_event_end,
            rolling_window=rolling_window,
            confidence_level=confidence_level,
            num_scenarios=1000  # Optional: explicitly specify number of scenarios
        )
                
        # Create COVID analysis plots
        covid_plot_path = plot_stress_period_analysis_covid(
            covid_results_df,
            covid_stress_stats,
            portfolio_name,
            plots_dir
        )
        
        
        dist_plot_path = plot_return_distributions(
            portfolio_name, 
            predicted_returns,
            plot_returns,
            plots_dir
        )
        print(f"Saved distribution analysis to: {dist_plot_path}")
        
        # Save results
        portfolio_id = portfolio_name.replace(" ", "_").replace("(", "").replace(")", "")
        
        analysis_path = os.path.join(results_dir, f"{portfolio_id}_analysis.csv")
        metrics_path = os.path.join(results_dir, f"{portfolio_id}_risk_metrics.csv")
        # Save files
        risk_metrics_df.to_csv(metrics_path, index=False)
        with open(analysis_path, 'w') as f:
            f.write(f"Portfolio: {portfolio_name}\n\n")
            f.write("Portfolio Statistics:\n")
            f.write(portfolio_stats.to_string())
            f.write("\n\nRisk Metrics Summary:\n")
            f.write(summary_stats.to_string())
            f.write("\n\nPortfolio Composition:\n")
            f.write(weights_df.to_string())
        
        print(f"Saved portfolio analysis to: {analysis_path}")
        print(f"Saved daily risk metrics to: {metrics_path}")
        
        # Save COVID analysis results
        covid_results_path = os.path.join(covid_analysis_dir, 
                                        f"{portfolio_name}_covid_analysis.csv")
        covid_stats_path = os.path.join(covid_analysis_dir, 
                                      f"{portfolio_name}_covid_stats.csv")
        
        # Save COVID analysis files
        covid_results_df.to_csv(covid_results_path, index=False)
        pd.Series(covid_stress_stats).to_csv(covid_stats_path)
        
        # Create comparative analysis between normal period and stress period
        comparative_stats = pd.DataFrame({
            'Metric': [
                'VaR_Breach_Rate',
                'Avg_Historical_VaR',
                'Avg_LSTM_VaR',
                'Avg_MC_VaR',
                'Returns_Volatility',
                'Max_Loss'
            ],
            'Normal_Period': [
                summary_stats['Historical_VaR_Breach_Rate'],
                summary_stats['Avg_Historical_VaR'],
                summary_stats['Avg_LSTM_VaR'],
                summary_stats['Avg_MC_VaR'],
                summary_stats['Actual_Returns_Std'],
                portfolio_stats['Max_Drawdown']
            ],
            'Stress_Period': [
                covid_stress_stats['Historical_VaR_Breach_Rate'],
                covid_stress_stats['Stress_Historical_VaR_Mean'],
                covid_stress_stats['Stress_LSTM_VaR_Mean'],
                covid_stress_stats['Stress_MC_VaR_Mean'],
                covid_stress_stats['Stress_Period_Returns_Std'],
                covid_stress_stats['Stress_Period_Worst_Loss']
            ]
        })
        
        comparative_stats['Change_Pct'] = (
            (comparative_stats['Stress_Period'] / comparative_stats['Normal_Period'] - 1) * 100
        )
        
        # Save comparative analysis
        comparative_path = os.path.join(covid_analysis_dir, 
                                      f"{portfolio_name}_comparative_analysis.csv")
        comparative_stats.to_csv(comparative_path, index=False)
        
        print(f"Saved COVID-19 analysis results to: {covid_results_path}")
        print(f"Saved COVID-19 statistics to: {covid_stats_path}")
        print(f"Saved comparative analysis to: {comparative_path}")
        print(f"Saved COVID-19 analysis plots to: {covid_plot_path}")
        
        # Print key findings
        print("\nKey COVID-19 Period Findings:")
        print(f"Historical VaR Breach Rate: {covid_stress_stats['Historical_VaR_Breach_Rate']:.2f}%")
        print(f"LSTM VaR Breach Rate: {covid_stress_stats['LSTM_VaR_Breach_Rate']:.2f}%")
        print(f"Worst Daily Loss: {covid_stress_stats['Stress_Period_Worst_Loss']:.2%}")
        print(f"Historical VaR Change: {covid_stress_stats['Historical_VaR_Change_Pct']:.2f}%")
        print(f"LSTM VaR Change: {covid_stress_stats['LSTM_VaR_Change_Pct']:.2f}%")
        print("\nMonte Carlo VaR Statistics:")
        print(f"MC VaR Breach Rate: {mc_var_breaches.mean() * 100:.2f}%")
        print(f"Average MC VaR: {mc_vars.mean():.6f}")
        print(f"MC VaR Coverage Ratio: {(mc_var_breaches.mean() * 100) / 5.0:.2f}")


if __name__ == "__main__":
    main()