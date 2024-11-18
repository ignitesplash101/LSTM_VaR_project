import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir)

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

def download_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date, group_by="column")
    return data['Adj Close']

def download_sp500_index(start_date, end_date):
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    return sp500['Adj Close']

def clean_and_filter_data(df, min_pct_non_null=0.95, min_history_years=5):
    print("Initial shape:", df.shape)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    min_non_null_count = int(len(df) * min_pct_non_null)
    df = df.dropna(axis=1, thresh=min_non_null_count)
    print(f"Shape after dropping columns with less than {min_pct_non_null*100}% non-null values:", df.shape)

    min_history_date = df.index.max() - pd.Timedelta(days=365*min_history_years)
    enough_history = df.apply(lambda col: col.first_valid_index() <= min_history_date)
    df = df.loc[:, enough_history]
    print(f"Shape after dropping columns with less than {min_history_years} years of history:", df.shape)

    return df

def calculate_daily_returns(df):
    return df.pct_change()

def impute_missing_values(returns):
    imputer = SimpleImputer(strategy='mean')
    imputed_returns = pd.DataFrame(imputer.fit_transform(returns), columns=returns.columns, index=returns.index)
    return imputed_returns

def perform_pca(returns, n_components=1):
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_returns)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca, pca_result

def create_pca_portfolio(pca, returns):
    weights = pca.components_[0]  # Use the first principal component as weights
    weights = weights / np.sum(np.abs(weights))  # Normalize weights
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def plot_returns_histogram(returns, ticker):
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, kde=True, stat="density")
    plt.title(f'Distribution of Daily Returns Over 10 Years - {ticker}')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    filename = os.path.join(script_dir, f'{ticker}_returns_histogram.png')
    plt.savefig(filename)
    plt.close()
    print(f"Histogram of returns saved as '{filename}'")

def plot_cumulative_returns(pc1_cumulative_returns, sp500_cumulative_returns):
    plt.figure(figsize=(12, 6))
    
    # Convert to percentage returns
    pc1_pct_returns = (pc1_cumulative_returns - 1) * 100
    sp500_pct_returns = (sp500_cumulative_returns - 1) * 100
    
    plt.plot(pc1_pct_returns.index, pc1_pct_returns.values, label='PC1 Portfolio')
    plt.plot(sp500_pct_returns.index, sp500_pct_returns.values, label='S&P 500 Index')
    plt.title('Cumulative Returns of extracted PC1 vs the S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True)
    plt.legend()

    # Improve x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Improve y-axis
    def y_fmt(y, pos):
        return f'{y:.0f}%'

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    # Add horizontal lines at key percentage levels
    for y in [0, 100, 200, 400]:
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)

    # Set y-axis limits to ensure 0% is visible
    plt.ylim(bottom=min(pc1_pct_returns.min(), sp500_pct_returns.min(), 0))

    filename = os.path.join(script_dir, 'pc1_vs_sp500_cumulative_returns_percent.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative returns comparison plot (in percentages) saved as '{filename}'")
    
def main():
    try:
        os.chdir(script_dir)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)  # 10 years ago
        
        print("Fetching S&P 500 symbols...")
        symbols = get_sp500_symbols()
        
        print(f"Downloading data for {len(symbols)} stocks...")
        df = download_stock_data(symbols, start_date, end_date)
        
        print("Downloading S&P 500 index data...")
        sp500_index = download_sp500_index(start_date, end_date)
        
        print("Cleaning and filtering data...")
        df_cleaned = clean_and_filter_data(df, min_pct_non_null=0.95, min_history_years=5)
        
        print("Calculating daily returns...")
        df_returns = calculate_daily_returns(df_cleaned)
        sp500_returns = calculate_daily_returns(sp500_index)
        
        print("Imputing missing values...")
        df_returns_imputed = impute_missing_values(df_returns)
        
        print(f"Final data shape: {df_returns_imputed.shape}")
        print(f"Date range: {df_returns_imputed.index.min()} to {df_returns_imputed.index.max()}")
        
        # Save CSV files
        df_cleaned.to_csv(os.path.join(script_dir, "sp500_adjusted_close_cleaned.csv"))
        df_returns_imputed.to_csv(os.path.join(script_dir, "sp500_daily_returns_imputed.csv"))
        
        # Plot histogram for Microsoft
        msft_returns = df_returns_imputed['MSFT'].dropna()
        plot_returns_histogram(msft_returns, "MSFT")
        
        # Perform PCA
        pca, pca_result = perform_pca(df_returns_imputed, n_components=1)
        
        # Create PCA-based portfolio (PC1)
        pc1_portfolio_returns = create_pca_portfolio(pca, df_returns_imputed)
        pc1_cumulative_returns = (1 + pc1_portfolio_returns).cumprod()
        
        # Calculate S&P 500 cumulative returns
        sp500_cumulative_returns = (1 + sp500_returns).cumprod()
        
        # Plot comparison of returns
        plot_cumulative_returns(pc1_cumulative_returns, sp500_cumulative_returns)
        
        # Save PC1 and S&P 500 returns to CSV
        pd.DataFrame({
            'PC1_Returns': pc1_portfolio_returns,
            'SP500_Returns': sp500_returns
        }).to_csv(os.path.join(script_dir, "pc1_vs_sp500_returns.csv"))
        
        print("Analysis complete. Check the generated PNG files for visualizations and CSV files for data.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()