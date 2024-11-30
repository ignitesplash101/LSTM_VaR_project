import os
import numpy as np
import pandas as pd
import torch
import random
from scipy import stats
from data_processing import create_pca_portfolios, calculate_portfolio_returns
from portfolio_analysis import analyze_portfolio_var, analyze_covid_stress_period
from visualization import plot_return_distributions, plot_risk_measures_comparison, plot_stress_period_analysis_covid, plot_lstm_analysis

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Set random seeds for reproducibility
    set_seeds()
    
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
    backtest_days = 378  # Approximately 1.5 years of trading days
    rolling_window = 1008  # 4-year rolling window
    confidence_level = 0.95
    num_scenarios = 1000
    
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
        
        # Calculate portfolio returns and risk metrics
        predicted_returns, hist_vars, hist_es, lstm_vars, lstm_es, \
        mc_vars, mc_es, hist_es_breaches, lstm_es_breaches, \
        mc_es_breaches, hist_var_breaches, lstm_var_breaches, \
        mc_var_breaches, model, X_test, scaler = analyze_portfolio_var(
            returns_df, weights, 
            backtest_days=backtest_days,
            confidence_level=confidence_level, 
            rolling_window=rolling_window,
            num_scenarios=num_scenarios
        )
        
        # Prepare data for plotting
        plot_dates = dates[-backtest_days:]
        plot_returns = calculate_portfolio_returns(returns_df, weights)[-backtest_days:]
        
        # Align all series lengths
        min_len = min(len(plot_dates), len(plot_returns), len(hist_vars), 
                     len(hist_es), len(predicted_returns))
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
        mse_predicted_returns = np.mean((predicted_returns - plot_returns) ** 2)

        # Calculate portfolio statistics
        portfolio_stats = pd.Series({
            'Annual_Return': np.mean(plot_returns) * 252,
            'Annual_Volatility': np.std(plot_returns) * np.sqrt(252),
            'Sharpe_Ratio': np.mean(plot_returns) / np.std(plot_returns) * np.sqrt(252),
            'Historical_VaR_Breach_Rate': hist_var_breaches.mean() * 100,
            'LSTM_VaR_Breach_Rate': lstm_var_breaches.mean() * 100,
            'MonteCarlo_VaR_Breach_Rate': mc_var_breaches.mean() * 100,
            'Skewness': stats.skew(plot_returns),
            'Excess_Kurtosis': stats.kurtosis(plot_returns),
            'Max_Drawdown': np.min(plot_returns),
            'VaR_95_Historical': np.percentile(plot_returns, 5)
        })
        
        # Calculate summary statistics
        summary_stats = pd.Series({
            'Historical_VaR_Breach_Rate': hist_var_breaches.mean() * 100,
            'LSTM_VaR_Breach_Rate': lstm_var_breaches.mean() * 100,
            'MonteCarlo_VaR_Breach_Rate': mc_var_breaches.mean() * 100,
            'Avg_Historical_VaR': hist_vars.mean(),
            'Avg_Historical_ES': hist_es.mean(),
            'Avg_LSTM_VaR': lstm_vars.mean(),
            'Avg_LSTM_ES': lstm_es.mean(),
            'Avg_MonteCarlo_VaR': mc_vars.mean(),
            'Avg_MonteCarlo_ES': mc_es.mean(),
            'LSTM_Prediction_MAE': np.mean(np.abs(plot_returns - predicted_returns)),
            'LSTM_Prediction_MSE': mse_predicted_returns,
            'LSTM_Prediction_Std': np.std(predicted_returns),
            'Actual_Returns_Std': np.std(plot_returns),
            'LSTM_VaR_Coverage_Ratio': (lstm_var_breaches.mean() * 100) / 5.0,
            'MonteCarlo_VaR_Coverage_Ratio': (mc_var_breaches.mean() * 100) / 5.0,
            'Historical_VaR_Coverage_Ratio': (hist_var_breaches.mean() * 100) / 5.0,
            'LSTM_ES_vs_Historical_ES_Ratio': lstm_es.mean() / hist_es.mean(),
            'MonteCarlo_ES_vs_Historical_ES_Ratio': mc_es.mean() / hist_es.mean()
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

        # Save metrics to DataFrame, including breaches
        risk_metrics_df = pd.DataFrame({
            'Date': plot_dates,
            'Portfolio': portfolio_name,
            'Actual_Returns': plot_returns,
            'LSTM_Predicted_Returns': predicted_returns,
            'Historical_VaR': hist_vars,
            'Historical_ES': hist_es,
            'LSTM_VaR': lstm_vars,
            'LSTM_ES': lstm_es,
            'MonteCarlo_VaR': mc_vars,
            'MonteCarlo_ES': mc_es,
            'Historical_VaR_Breach': hist_var_breaches.astype(int),
            'LSTM_VaR_Breach': lstm_var_breaches.astype(int),
            'MonteCarlo_VaR_Breach': mc_var_breaches.astype(int),
            'Historical_ES_Breach': hist_es_breaches.astype(int),
            'LSTM_ES_Breach': lstm_es_breaches.astype(int),
            'MonteCarlo_ES_Breach': mc_es_breaches.astype(int)
        })

        # Add breach rates as summary rows at the bottom
        breach_rates = pd.DataFrame({
            'Date': ['Breach Rates'],
            'Portfolio': [portfolio_name],
            'Historical_VaR_Breach': [np.mean(hist_var_breaches) * 100],
            'LSTM_VaR_Breach': [np.mean(lstm_var_breaches) * 100],
            'MonteCarlo_VaR_Breach': [np.mean(mc_var_breaches) * 100],
            'Historical_ES_Breach': [np.mean(hist_es_breaches) * 100],
            'LSTM_ES_Breach': [np.mean(lstm_es_breaches) * 100],
            'MonteCarlo_ES_Breach': [np.mean(mc_es_breaches) * 100]
        })

        risk_metrics_df = pd.concat([risk_metrics_df, breach_rates], ignore_index=True)

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

        # Create plots
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
            lstm_es_breaches,  # Add lstm_es_breaches
            mc_es_breaches,    # Add mc_es_breaches
            hist_var_breaches, # Add hist_var_breaches
            lstm_var_breaches, # Add lstm_var_breaches
            mc_var_breaches,   # Add mc_var_breaches
            plots_dir          # Add plots_dir
        )
        print(f"Saved risk measures comparison plot to: {risk_plot_path}")

        dist_plot_path = plot_return_distributions(
            portfolio_name, 
            predicted_returns,
            plot_returns,
            plots_dir
        )
        print(f"Saved distribution analysis to: {dist_plot_path}")

        lstm_analysis_path = plot_lstm_analysis(
            model,  # You'll need to pass the trained model from analyze_portfolio_var
            X_test,  # Pass the test sequences from analyze_portfolio_var
            plot_returns,
            plot_dates,
            portfolio_name,
            plots_dir
        )
        print(f"Saved LSTM analysis to: {lstm_analysis_path}")
        
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
                'Avg_MonteCarlo_VaR',
                'Returns_Volatility',
                'Max_Loss'
            ],
            'Normal_Period': [
                summary_stats['Historical_VaR_Breach_Rate'],
                summary_stats['Avg_Historical_VaR'],
                summary_stats['Avg_LSTM_VaR'],
                mc_vars.mean(),
                summary_stats['Actual_Returns_Std'],
                portfolio_stats['Max_Drawdown']
            ],
            'Stress_Period': [
                covid_stress_stats['Historical_VaR_Breach_Rate'],
                covid_stress_stats['Stress_Historical_VaR_Mean'],
                covid_stress_stats['Stress_LSTM_VaR_Mean'],
                covid_stress_stats['Stress_MonteCarlo_VaR_Mean'],
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
        print(f"Monte Carlo VaR Breach Rate: {covid_stress_stats['MonteCarlo_VaR_Breach_Rate']:.2f}%")
        print(f"Worst Daily Loss: {covid_stress_stats['Stress_Period_Worst_Loss']:.2%}")
        print(f"Historical VaR Change: {covid_stress_stats['Historical_VaR_Change_Pct']:.2f}%")
        print(f"LSTM VaR Change: {covid_stress_stats['LSTM_VaR_Change_Pct']:.2f}%")
        print(f"Monte Carlo VaR Change: {covid_stress_stats['MonteCarlo_VaR_Change_Pct']:.2f}%")



if __name__ == "__main__":
    main()