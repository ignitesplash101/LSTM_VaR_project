import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import time
import random
from requests.exceptions import RequestException

def get_sp500_symbols():
    """
    Get S&P 500 symbols and their sectors from Wikipedia
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df[['Symbol', 'GICS Sector']]  # Get both symbol and sector

def get_stock_info_with_retry(symbol, max_retries=3, initial_delay=1):
    """
    Get stock info with retry logic and exponential backoff
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # Add random delay to avoid synchronized requests
            time.sleep(delay + random.uniform(0, 1))
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'Symbol': symbol,
                'MarketCap': info.get('marketCap', None),
                'Sector': info.get('sector', None),
                'Industry': info.get('industry', None),
                'SubIndustry': info.get('subIndustry', None)
            }
            
        except Exception as e:
            if "Too Many Requests" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    delay *= 2  # Exponential backoff
                    print(f"Rate limit hit for {symbol}, retrying in {delay} seconds...")
                    continue
            print(f"Error fetching data for {symbol}: {str(e)}")
            return {
                'Symbol': symbol,
                'MarketCap': None,
                'Sector': None,
                'Industry': None,
                'SubIndustry': None
            }
    
    return None

def get_stock_info(symbols, batch_size=10):
    """
    Get detailed stock information including market cap and sector
    """
    stock_info = []
    total = len(symbols)
    
    print("Fetching stock information...")
    
    # Process symbols in batches
    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        
        for symbol in batch:
            info = get_stock_info_with_retry(symbol)
            if info:
                stock_info.append(info)
        
        # Progress update
        print(f"Processed {min(i + batch_size, total)}/{total} stocks")
        
        # Add delay between batches
        if i + batch_size < total:
            time.sleep(2)  # 2 second delay between batches
    
    return pd.DataFrame(stock_info)

def download_stock_data_with_retry(symbols, start_date, end_date, max_retries=3):
    """
    Download stock data with retry logic
    """
    for attempt in range(max_retries):
        try:
            return yf.download(symbols, start=start_date, end=end_date, group_by="column")['Adj Close']
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff
                print(f"Error downloading data, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise e

def clean_and_filter_data(df, min_pct_non_null=0.95, min_history_years=5):
    print("Initial shape:", df.shape)

    # Replace any infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns with too many NaN values
    min_non_null_count = int(len(df) * min_pct_non_null)
    df = df.dropna(axis=1, thresh=min_non_null_count)
    print(f"Shape after dropping columns with less than {min_pct_non_null*100}% non-null values:", df.shape)

    # Drop columns (stocks) that don't have enough history
    min_history_date = df.index.max() - pd.Timedelta(days=365*min_history_years)
    enough_history = df.apply(lambda col: col.first_valid_index() <= min_history_date)
    df = df.loc[:, enough_history]
    print(f"Shape after dropping columns with less than {min_history_years} years of history:", df.shape)

    # Forward fill remaining NaN values
    df.fillna(method='ffill', inplace=True)
    
    # Backward fill any remaining NaN values at the beginning
    df.fillna(method='bfill', inplace=True)

    return df

def calculate_daily_returns(df):
    return df.pct_change()

def validate_and_clean_stock_info(stock_info_df):
    """
    Validate and clean stock information, filling missing sectors with Wikipedia data
    """
    # Get Wikipedia sector information as backup
    wiki_sectors = get_sp500_symbols()
    wiki_sectors = wiki_sectors.set_index('Symbol')['GICS Sector']
    
    # Fill missing sectors with Wikipedia data
    missing_sectors = stock_info_df['Sector'].isna()
    if missing_sectors.any():
        print(f"Filling {missing_sectors.sum()} missing sectors with Wikipedia data")
        for symbol in stock_info_df[missing_sectors]['Symbol']:
            if symbol in wiki_sectors.index:
                stock_info_df.loc[stock_info_df['Symbol'] == symbol, 'Sector'] = wiki_sectors[symbol]
    
    # Fill any remaining missing sectors
    stock_info_df['Sector'] = stock_info_df['Sector'].fillna('Other')
    stock_info_df['Industry'] = stock_info_df['Industry'].fillna('Other')
    stock_info_df['SubIndustry'] = stock_info_df['SubIndustry'].fillna('Other')
    
    return stock_info_df

def save_market_caps(stock_info_df, data_dir):
    """
    Extract and save market cap information to a CSV file, sorted by symbol
    """
    market_caps = stock_info_df[['Symbol', 'MarketCap']].copy()
    # Sort by Symbol alphabetically
    market_caps = market_caps.sort_values('Symbol')
    # Drop any rows with null market caps
    market_caps = market_caps.dropna(subset=['MarketCap'])
    # Ensure MarketCap is integer
    market_caps['MarketCap'] = market_caps['MarketCap'].astype('Int64')
    # Save without index
    market_caps.to_csv(os.path.join(data_dir, "sp500_market_caps.csv"), index=False)
    print(f"Saved market cap data for {len(market_caps)} stocks")
    return market_caps

def main():
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*15)  # 15 years ago
        
        print("Fetching S&P 500 symbols...")
        sp500_data = get_sp500_symbols()
        symbols = sp500_data['Symbol'].tolist()
        
        print("Fetching detailed stock information...")
        stock_info_df = get_stock_info(symbols)
        stock_info_df = validate_and_clean_stock_info(stock_info_df)
        
        # Save market caps before filtering
        market_caps = save_market_caps(stock_info_df, data_dir)
        
        print(f"Downloading price data for {len(symbols)} stocks...")
        df = download_stock_data_with_retry(symbols, start_date, end_date)
        
        print("Cleaning and filtering data...")
        df_cleaned = clean_and_filter_data(df, min_pct_non_null=0.95, min_history_years=5)
        
        print("Calculating daily returns...")
        df_returns = calculate_daily_returns(df_cleaned)
        
        # Filter stock info to match cleaned data
        cleaned_symbols = df_cleaned.columns.tolist()
        stock_info_df = stock_info_df[stock_info_df['Symbol'].isin(cleaned_symbols)]
        
        print(f"\nSector distribution:")
        print(stock_info_df['Sector'].value_counts())
        
        # Save files
        df_cleaned.reset_index().to_csv(os.path.join(data_dir, "sp500_adjusted_close_cleaned.csv"), index=False)
        df_returns.iloc[1:].reset_index().to_csv(os.path.join(data_dir, "sp500_daily_returns.csv"), index=False)
        stock_info_df.to_csv(os.path.join(data_dir, "sp500_stock_info.csv"), index=False)
        
        print("\nFiles saved successfully. Data summary:")
        print(f"Price data shape: {df_cleaned.shape}")
        print(f"Returns data shape: {df_returns.shape}")
        print(f"Stock info shape: {stock_info_df.shape}")
        print(f"Market caps saved: {len(market_caps)} stocks")
        print(f"Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
        
        # Display sample of stock info
        print("\nSample stock information:")
        print(stock_info_df[['Symbol', 'Sector', 'Industry']].head())
        
        print(f"\nFiles saved to {data_dir}/:")
        print("- sp500_adjusted_close_cleaned.csv")
        print("- sp500_daily_returns.csv")
        print("- sp500_stock_info.csv")
        print("- sp500_market_caps.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()