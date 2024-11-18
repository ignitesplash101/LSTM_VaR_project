import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from pathlib import Path

def load_all_portfolio_data(results_dir):
    """
    Load and parse all portfolio analysis files from the results directory
    """
    portfolio_data = {}
    files = Path(results_dir).glob("*_analysis.csv")
    
    for file_path in files:
        portfolio_id = file_path.stem.replace('_analysis', '')
        print(f"\nProcessing {portfolio_id}...")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Initialize data structures
        current_section = None
        portfolio_stats = {}
        risk_metrics = {}
        portfolio_composition = []
        header_found = False
        
        for line in lines:
            line = line.strip()
            
            if 'Portfolio:' in line:
                portfolio_name = line.split('Portfolio:')[1].strip()
            elif 'Portfolio Statistics:' in line:
                current_section = 'stats'
            elif 'Risk Metrics Summary:' in line:
                current_section = 'risk'
            elif 'Portfolio Composition:' in line:
                current_section = 'composition'
                header_found = False
            elif line and current_section:
                if current_section in ['stats', 'risk']:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            metric = parts[0]
                            value = float(parts[1])
                            if current_section == 'stats':
                                portfolio_stats[metric] = value
                            else:
                                risk_metrics[metric] = value
                    except ValueError:
                        continue
                elif current_section == 'composition':
                    if 'Stock' in line:
                        header_found = True
                        continue
                    
                    if header_found and line:
                        try:
                            # Line format:
                            # Symbol Symbol Weight Risk_Contribution Expected_Shortfall_Contribution Weight_Percentage
                            parts = line.split()
                            if len(parts) >= 6:  # Changed from 5 to 6
                                portfolio_composition.append({
                                    'Symbol': parts[0],  # Using 'Symbol' instead of 'stock'
                                    'Weight': float(parts[2]),  # Changed index from 1 to 2
                                    'Weight_Percentage': float(parts[-1])  # Using original column name
                                })
                        except ValueError as e:
                            continue
        
        print(f"\nFound {len(portfolio_composition)} stocks in portfolio")
        
        portfolio_data[portfolio_id] = {
            'name': portfolio_name,
            'stats': portfolio_stats,
            'risk_metrics': risk_metrics,
            'composition': pd.DataFrame(portfolio_composition)
        }
        
        print(f"Composition DataFrame shape: {portfolio_data[portfolio_id]['composition'].shape}")
        if not portfolio_composition:
            print("WARNING: No portfolio composition data found!")
        
    return portfolio_data


def load_all_risk_metrics(results_dir):
    """
    Load all daily risk metrics files with improved error handling and validation.
    
    Args:
        results_dir (str): Path to the directory containing risk metrics files.
    
    Returns:
        dict: Dictionary where keys are portfolio IDs and values are DataFrames with metrics data.
    """
    metrics_data = {}
    files = Path(results_dir).glob("*_risk_metrics.csv")
    
    for file_path in files:
        portfolio_id = file_path.stem.replace('_risk_metrics', '')
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Validate presence of the 'Date' column
            if 'Date' not in df.columns:
                print(f"WARNING: Missing 'Date' column in {file_path.name}. Skipping file.")
                continue
            
            # Convert 'Date' column to datetime, coercing invalid values to NaT
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Drop rows with invalid or missing dates
            if df['Date'].isna().any():
                invalid_count = df['Date'].isna().sum()
                print(f"WARNING: Dropping {invalid_count} rows with invalid dates in {file_path.name}")
                df = df.dropna(subset=['Date'])
            
            # Store the cleaned DataFrame in the dictionary
            metrics_data[portfolio_id] = df
        
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    return metrics_data


def analyze_sector_exposures(portfolio_data, stock_info_df):
    """
    Analyze sector exposures using the detailed stock information
    """
    # Initialize DataFrame to store sector exposures
    all_sectors = stock_info_df['Sector'].unique()
    sector_exposures = pd.DataFrame(0, 
                                  index=all_sectors,
                                  columns=portfolio_data.keys())
    
    # Calculate sector exposures for each portfolio
    for portfolio_id, data in portfolio_data.items():
        portfolio_weights = data['composition']
        
        # Merge weights with sector information
        portfolio_sectors = pd.merge(portfolio_weights, 
                                   stock_info_df[['Symbol', 'Sector', 'Industry']],
                                   on='Symbol',
                                   how='left')
        
        # Group by sector and sum weights
        sector_weights = portfolio_sectors.groupby('Sector')['Weight_Percentage'].sum()
        sector_exposures[portfolio_id] = sector_weights
    
    return sector_exposures

def calculate_correlation_matrix(metrics_data):
    """
    Calculate return correlations between portfolios
    """
    # Combine returns from all portfolios
    returns_data = pd.DataFrame()
    
    for portfolio_id, metrics in metrics_data.items():
        returns_data[portfolio_id] = metrics['Actual_Returns']
    
    return returns_data.corr()

def analyze_risk_characteristics(portfolio_data, metrics_data):
    """
    Analyze detailed risk characteristics of each portfolio
    """
    risk_characteristics = {}
    
    for portfolio_id, data in portfolio_data.items():
        returns = metrics_data[portfolio_id]['Actual_Returns']
        
        # Calculate various risk metrics
        characteristics = {
            'Daily_VaR_95': np.percentile(returns, 5),
            'Daily_CVaR_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'Max_Drawdown': data['stats']['Max_Drawdown'],
            'Downside_Volatility': returns[returns < 0].std() * np.sqrt(252),
            'Skewness': stats.skew(returns),
            'Excess_Kurtosis': stats.kurtosis(returns),
            'Sharpe_Ratio': data['stats']['Sharpe_Ratio'],
            'VaR_Breach_Rate': data['stats']['Historical_VaR_Breach_Rate'],
            'LSTM_Accuracy': 1 - data['stats']['LSTM_VaR_Breach_Rate'] / 5  # Relative to 5% target
        }
        
        risk_characteristics[portfolio_id] = characteristics
    
    return pd.DataFrame(risk_characteristics).T

def plot_combined_metrics(portfolio_data, metrics_data, stock_info_df, output_dir):
    """
    Create a combined multi-panel plot for portfolio correlations, sector weights, and cumulative returns.

    Args:
        portfolio_data (dict): Portfolio compositions.
        metrics_data (dict): Risk metrics data.
        stock_info_df (pd.DataFrame): Stock information with sector details.
        output_dir (str): Directory to save the plot.
    """
    import matplotlib.gridspec as gridspec

    # Create figure and grid layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.5])
    
    # === Portfolio Correlations Heatmap ===
    ax1 = fig.add_subplot(gs[0, 0])
    correlation_matrix = calculate_correlation_matrix(metrics_data)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1, fmt=".2f")
    ax1.set_title('Portfolio Correlations', fontsize=14)
    ax1.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax1.set_yticklabels(correlation_matrix.columns, rotation=0)

    # === Sector Weights Heatmap ===
    ax2 = fig.add_subplot(gs[0, 1])
    sector_exposures = analyze_sector_exposures(portfolio_data, stock_info_df)
    sns.heatmap(sector_exposures, annot=True, cmap='YlGnBu', ax=ax2, fmt=".1f", cbar=True)
    ax2.set_title('Sector Weights by Portfolio', fontsize=14)
    ax2.set_xlabel('Portfolios')
    ax2.set_ylabel('Sectors')

    # === Cumulative Returns Plot ===
    ax3 = fig.add_subplot(gs[1, :])
    for portfolio_id, metrics in metrics_data.items():
        cumulative_returns = (1 + metrics['Actual_Returns']).cumprod()
        ax3.plot(metrics['Date'], cumulative_returns, label=portfolio_id)
    ax3.set_title('Cumulative Returns', fontsize=14)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Return')
    ax3.legend(loc='upper left')
    ax3.grid(linestyle='--', alpha=0.7)

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Combined plot saved to {plot_path}")

    
def plot_sector_exposures(portfolio_data, stock_info_df, output_dir):
    """
    Plot Sector Exposures Heatmap
    """
    sector_exposures = analyze_sector_exposures(portfolio_data, stock_info_df)
    plt.figure(figsize=(12, 8))
    sns.heatmap(sector_exposures, annot=True, fmt='.1f', cmap='YlOrRd', cbar=True)
    plt.title('Sector Exposures (%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sector_exposures.png"), dpi=300)
    plt.close()
    
def plot_portfolio_correlations(metrics_data, output_dir):
    """
    Plot Portfolio Correlations Heatmap
    """
    correlations = calculate_correlation_matrix(metrics_data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title('Portfolio Correlations', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "portfolio_correlations.png"), dpi=300)
    plt.close()  

def plot_cumulative_returns(metrics_data, output_dir):
    """
    Plot Cumulative Returns
    """
    plt.figure(figsize=(12, 6))
    for portfolio_id, metrics in metrics_data.items():
        cumulative_returns = (1 + metrics['Actual_Returns']).cumprod()
        plt.plot(metrics['Date'], cumulative_returns, label=portfolio_id)
    plt.title('Cumulative Returns', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_returns.png"), dpi=300)
    plt.close()
    
def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    data_dir = os.path.join(current_dir, "data", "data")
    
    print(f"Looking for data in: {data_dir}")
    print(f"Looking for results in: {results_dir}")
    
    # Load data
    print("Loading portfolio data...")
    portfolio_data = load_all_portfolio_data(results_dir)
    
    print("Loading risk metrics...")
    metrics_data = load_all_risk_metrics(results_dir)
    
    print("Loading stock information...")
    stock_info_path = os.path.join(data_dir, "sp500_stock_info.csv")
    if not os.path.exists(stock_info_path):
        raise FileNotFoundError(f"Stock info file not found at: {stock_info_path}")
    stock_info_df = pd.read_csv(stock_info_path)
    
    # Create output directory
    output_dir = os.path.join(results_dir, "pca_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate combined plot
    print("Generating combined metrics plot...")
    plot_combined_metrics(portfolio_data, metrics_data, stock_info_df, output_dir)
    print("Plots generated successfully.")

if __name__ == "__main__":
    main()
