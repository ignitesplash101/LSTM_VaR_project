import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir)

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

def get_sp500_symbols():
    """
    Get S&P 500 symbols and their sectors from Wikipedia
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df[['Symbol', 'GICS Sector']]  # Get both symbol and sector

def get_stock_info(symbols):
    """
    Get detailed stock information including market cap and sector
    """
    stock_info = []
    total = len(symbols)
    
    print("Fetching stock information...")
    for i, symbol in enumerate(symbols):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            stock_data = {
                'Symbol': symbol,
                'MarketCap': info.get('marketCap', None),
                'Sector': info.get('sector', None),
                'Industry': info.get('industry', None),
                'SubIndustry': info.get('subIndustry', None)
            }
            stock_info.append(stock_data)
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{total} stocks")
                
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    return pd.DataFrame(stock_info)

def download_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date, group_by="column")
    return data['Adj Close']

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
        
        print(f"Downloading price data for {len(symbols)} stocks...")
        df = download_stock_data(symbols, start_date, end_date)
        
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
        print(f"Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
        
        # Display sample of stock info
        print("\nSample stock information:")
        print(stock_info_df[['Symbol', 'Sector', 'Industry']].head())
        
        print(f"\nFiles saved to {data_dir}/:")
        print("- sp500_adjusted_close_cleaned.csv")
        print("- sp500_daily_returns.csv")
        print("- sp500_stock_info.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()