import os 

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", 'data')
    data_path = os.path.join(data_dir, "sp500_adjusted_close_cleaned.csv")
    
    print("Current directory:", current_dir)
    print("Data directory:", data_dir)
    print("Looking for file:", data_path)
    print("File exists?", os.path.exists(data_path))
    
if __name__ == "__main__":
    main()