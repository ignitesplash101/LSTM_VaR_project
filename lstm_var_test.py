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
import torch.nn.functional as F  # Add this import at the top


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Prediction heads for mean and volatility
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Constrain returns to [-1, 1]
        )
        
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive values
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Scale predictions to match financial return magnitudes
        mean = self.mean_head(last_hidden) * 0.01  # Scaled to ~1% magnitude
        vol = self.vol_head(last_hidden) * 0.01
        return mean, vol

def create_sequences(data, sequence_length):
    """
    Create sequences from data for LSTM training.
    """
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(seq.reshape(-1, 1))  # Reshape for single feature
        targets.append(target)
    return np.array(sequences), np.array(targets).reshape(-1, 1)


def create_pca_portfolios(returns_df, market_caps_df=None, n_components=5):
    """
    Generate portfolio weights: equal-weight, market-cap-weight, PCA-based.
    """
    portfolios = []
    n_stocks = returns_df.shape[1]

    # Equal-weight portfolio
    equal_weight = np.ones(n_stocks) / n_stocks
    portfolios.append(equal_weight)
    
    # Market-cap-weight portfolio
    if market_caps_df is not None:
        market_caps = np.zeros(n_stocks)
        for i, symbol in enumerate(returns_df.columns):
            cap = market_caps_df[market_caps_df['Symbol'] == symbol]['MarketCap'].values
            if len(cap) > 0:
                market_caps[i] = cap[0]
        market_cap_weights = market_caps / np.sum(market_caps)
        portfolios.append(market_cap_weights)

    # PCA-based portfolios
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_df)
    pca = PCA(n_components=n_components)
    pca.fit(scaled_returns)
    for i in range(n_components):
        weights = np.abs(pca.components_[i])
        weights /= np.sum(weights)
        portfolios.append(weights)
    
    return portfolios


def calculate_rolling_var_es(returns, confidence_level=0.95, rolling_window=252):
    """
    Calculate daily rolling VaR and ES with proper daily updates.
    
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

def custom_finance_loss(pred_mean, pred_vol, targets, lambda_vol=0.1):
    """
    Custom loss function combining NLL and volatility penalties.
    """
    z = (targets - pred_mean) / (pred_vol + 1e-6)
    nll = 0.5 * (torch.log(pred_vol**2) + z**2)
    return_loss = F.mse_loss(pred_mean, targets)
    total_loss = return_loss + lambda_vol * nll.mean()
    return total_loss


def train_lstm_model(X_train, y_train, X_val, y_val, hidden_dim=50, num_layers=2, 
                     batch_size=32, patience=15, max_epochs=200):
    """
    Train LSTM model with early stopping and gradient clipping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]  # Input features
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_vol = model(X_batch)
            loss = custom_finance_loss(pred_mean, pred_vol, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                pred_mean, pred_vol = model(X_val_batch)
                val_loss += custom_finance_loss(pred_mean, pred_vol, y_val_batch).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss / len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(best_model_state)
    return model

def generate_scenarios(model, X_test, num_scenarios=1000):
    """
    Generate multiple return scenarios from the predicted distributions.
    """
    device = next(model.parameters()).device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        mean, std = model(X_test_tensor)
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        
    scenarios = []
    for i in range(num_scenarios):
        scenario = np.random.normal(mean, std)
        scenarios.append(scenario)
    
    return np.array(scenarios)

def analyze_portfolio_var(returns_df, weights, sequence_length=21, backtest_days=378,
                          confidence_level=0.95, rolling_window=252):
    """
    Analyze portfolio risk (VaR and ES) using LSTM predictions and historical metrics.
    """
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(returns_df, weights)
    
    # Split data for training and testing
    train_data = portfolio_returns[:-backtest_days]
    test_data = portfolio_returns[-backtest_days-sequence_length:]  # Include extra for first sequence

    # Prepare sequences for training
    X_train, y_train = create_sequences(train_data, sequence_length)

    # Split some validation data from training
    val_size = min(100, len(X_train) // 5)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    # Train the LSTM model
    print("Training LSTM model...")
    model = train_lstm_model(X_train, y_train, X_val, y_val, hidden_dim=50, num_layers=2)

    # Prepare sequences for testing
    X_test = []
    test_returns = []
    for i in range(len(test_data) - sequence_length):
        seq = test_data[i:i + sequence_length]
        next_return = test_data[i + sequence_length]
        X_test.append(seq.reshape(sequence_length, 1))
        test_returns.append(next_return)

    X_test = np.array(X_test)
    test_returns = np.array(test_returns)

    # Generate predictions
    print("Generating predictions...")
    predicted_returns, predicted_vols = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(len(X_test)):
            x_i = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
            pred_mean, pred_vol = model(x_i)
            predicted_returns.append(pred_mean.cpu().numpy()[0, 0])
            predicted_vols.append(pred_vol.cpu().numpy()[0, 0])

    predicted_returns = np.array(predicted_returns)
    predicted_vols = np.array(predicted_vols)

    # Calculate historical VaR/ES
    hist_vars, hist_es = [], []
    for t in range(len(test_returns) - rolling_window):
        window = test_returns[t:t + rolling_window]
        var = np.percentile(window, (1 - confidence_level) * 100)
        hist_vars.append(var)

        losses = window[window <= var]
        es = np.mean(losses) if len(losses) > 0 else var
        hist_es.append(es)

    # Calculate LSTM-based VaR and ES
    z_score = stats.norm.ppf(confidence_level)
    lstm_vars = -(predicted_returns + z_score * predicted_vols)
    es_factor = stats.norm.pdf(z_score) / (1 - confidence_level)
    lstm_es = -(predicted_returns + es_factor * predicted_vols)

    # Align series lengths
    n = min(len(predicted_returns), len(hist_vars))
    predicted_returns = predicted_returns[:n]
    lstm_vars = lstm_vars[:n]
    lstm_es = lstm_es[:n]
    hist_vars = hist_vars[:n]
    hist_es = hist_es[:n]
    actual_returns = test_returns[rolling_window:rolling_window + n]

    return actual_returns, hist_vars, hist_es, lstm_vars, lstm_es


def plot_risk_measures_comparison(portfolio_name, dates, returns, hist_vars, hist_es, 
                                  lstm_vars, lstm_es, plots_dir):
    """
    Compare actual returns, historical VaR/ES, and LSTM VaR/ES in a plot.
    """
    fig, axs = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)

    # Plot VaR comparison
    ax_var = axs[0]
    ax_var.plot(dates, returns, label="Actual Returns", color="blue", alpha=0.7)
    ax_var.plot(dates, hist_vars, label="Historical VaR", color="red", linestyle="--")
    ax_var.plot(dates, lstm_vars, label="LSTM VaR", color="green", linestyle="--")
    ax_var.axhline(0, color="gray", linestyle=":")
    ax_var.set_title(f"{portfolio_name}: Returns and VaR Measures")
    ax_var.set_xlabel("Date")
    ax_var.set_ylabel("Returns")
    ax_var.legend(loc='upper left')
    ax_var.grid(True)

    # Plot ES comparison
    ax_es = axs[1]
    ax_es.plot(dates, hist_es, label="Historical ES", color="red")
    ax_es.plot(dates, lstm_es, label="LSTM ES", color="green")
    ax_es.axhline(0, color="gray", linestyle=":")
    ax_es.set_title(f"{portfolio_name}: Expected Shortfall Measures")
    ax_es.set_xlabel("Date")
    ax_es.set_ylabel("Expected Shortfall")
    ax_es.legend(loc='upper left')
    ax_es.grid(True)

    save_path = os.path.join(plots_dir, f"{portfolio_name}_risk_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_return_distributions(portfolio_name, predicted_returns, actual_returns, plots_dir):
    """
    Plot return distributions: histograms and KDEs for predicted vs. actual.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(min(actual_returns.min(), predicted_returns.min()), 
                       max(actual_returns.max(), predicted_returns.max()), 50)

    ax.hist(predicted_returns, bins=bins, alpha=0.5, label="Predicted Returns", density=True)
    ax.hist(actual_returns, bins=bins, alpha=0.5, label="Actual Returns", density=True)

    kde_pred = gaussian_kde(predicted_returns)
    kde_act = gaussian_kde(actual_returns)
    x_range = np.linspace(min(bins), max(bins), 200)
    ax.plot(x_range, kde_pred(x_range), label="KDE Predicted", color="blue")
    ax.plot(x_range, kde_act(x_range), label="KDE Actual", color="green")

    ax.set_title(f"{portfolio_name}: Return Distributions")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True)

    save_path = os.path.join(plots_dir, f"{portfolio_name}_return_distributions.png")
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

    # Directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Parameters
    backtest_days = 378  # Approximately 1.5 years of trading days
    rolling_window = 252  # 1-year rolling window
    confidence_level = 0.95

    # Load data
    data_path = os.path.join(data_dir, "sp500_adjusted_close_cleaned.csv")
    market_caps_path = os.path.join(data_dir, "sp500_market_caps.csv")
    data = pd.read_csv(data_path)
    market_caps = pd.read_csv(market_caps_path)
    data['Date'] = pd.to_datetime(data['Date'])
    dates = data['Date'].values

    # Calculate returns
    log_prices = np.log(data.iloc[:, 1:])
    returns_df = log_prices.diff().dropna()
    dates = dates[1:]  # Align with returns

    # Create portfolios
    portfolios = create_pca_portfolios(returns_df, market_caps, n_components=5)

    # Analyze each portfolio
    for i, weights in enumerate(portfolios):
        portfolio_name = f"PCA Portfolio {i}" if i > 1 else ("Equal-Weight" if i == 0 else "Market-Cap-Weight")
        print(f"Analyzing {portfolio_name}...")

        # Perform risk analysis
        actual_returns, hist_vars, hist_es, lstm_vars, lstm_es = analyze_portfolio_var(
            returns_df, weights, sequence_length=21, backtest_days=backtest_days,
            confidence_level=confidence_level, rolling_window=rolling_window
        )

        # Align series for saving
        plot_dates = dates[-len(actual_returns):]
        result_df = pd.DataFrame({
            'Date': plot_dates,
            'Actual_Returns': actual_returns,
            'Historical_VaR': hist_vars,
            'Historical_ES': hist_es,
            'LSTM_VaR': lstm_vars,
            'LSTM_ES': lstm_es
        })

        # Save time-series data to CSV
        result_csv_path = os.path.join(results_dir, f"{portfolio_name}_risk_analysis.csv")
        result_df.to_csv(result_csv_path, index=False)
        print(f"Saved time-series data to: {result_csv_path}")

        # Generate and save plots
        risk_plot_path = plot_risk_measures_comparison(portfolio_name, plot_dates, actual_returns,
                                                       hist_vars, hist_es, lstm_vars, lstm_es, plots_dir)
        print(f"Saved risk comparison plot: {risk_plot_path}")

        dist_plot_path = plot_return_distributions(portfolio_name, lstm_vars, actual_returns, plots_dir)
        print(f"Saved distribution plot: {dist_plot_path}")



if __name__ == "__main__":
    main()