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

def plot_risk_measures_comparison(portfolio_id, dates, returns, hist_vars, hist_es, 
                                  lstm_vars, lstm_es, mc_vars, mc_es, 
                                  hist_es_breaches, lstm_es_breaches, mc_es_breaches,
                                  hist_var_breaches, lstm_var_breaches, mc_var_breaches, plots_dir):
    """
    Plot actual returns along with VaR and ES measures, including breach statistics and rates.
    """
    hist_var_breach_rate = np.mean(hist_var_breaches) * 100
    lstm_var_breach_rate = np.mean(lstm_var_breaches) * 100
    mc_var_breach_rate = np.mean(mc_var_breaches) * 100
    hist_es_breach_rate = np.mean(hist_es_breaches) * 100
    lstm_es_breach_rate = np.mean(lstm_es_breaches) * 100
    mc_es_breach_rate = np.mean(mc_es_breaches) * 100

    fig, axs = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)

    # Top Plot: Returns and VaR
    ax_var = axs[0]
    ax_var.plot(dates, returns, label="Actual Returns", color="blue", alpha=0.7)
    ax_var.plot(dates, hist_vars, label="Historical VaR", color="red", linestyle="--")
    ax_var.plot(dates, lstm_vars, label="LSTM VaR", color="green", linestyle="--")
    ax_var.plot(dates, mc_vars, label="Monte Carlo VaR", color="orange", linestyle="--")
    ax_var.scatter(dates[hist_var_breaches], returns[hist_var_breaches], color="red", marker="x", label="Hist VaR Breach")
    ax_var.scatter(dates[lstm_var_breaches], returns[lstm_var_breaches], color="green", marker="x", label="LSTM VaR Breach")
    ax_var.scatter(dates[mc_var_breaches], returns[mc_var_breaches], color="orange", marker="x", label="Monte Carlo VaR Breach")
    ax_var.axhline(0, color="gray", linestyle=":")

    breach_text = (
        f"Hist VaR Breach Rate: {hist_var_breach_rate:.2f}%\n"
        f"LSTM VaR Breach Rate: {lstm_var_breach_rate:.2f}%\n"
        f"Monte Carlo VaR Breach Rate: {mc_var_breach_rate:.2f}%"
    )
    ax_var.text(0.02, 0.98, breach_text, transform=ax_var.transAxes,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax_var.set_title(f"{portfolio_id}: Returns and VaR Measures (with Breaches)")
    ax_var.set_xlabel("Date")
    ax_var.set_ylabel("Returns")
    ax_var.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax_var.grid(True)

    # Bottom Plot: Returns and ES
    ax_es = axs[1]
    ax_es.plot(dates, returns, label="Actual Returns", color="blue", alpha=0.7)
    ax_es.plot(dates, hist_es, label="Historical ES", color="red", linestyle="--")
    ax_es.plot(dates, lstm_es, label="LSTM ES", color="green", linestyle="--")
    ax_es.plot(dates, mc_es, label="Monte Carlo ES", color="orange", linestyle="--")
    ax_es.scatter(dates[hist_es_breaches], returns[hist_es_breaches], color="red", marker="x", label="Hist ES Breach")
    ax_es.scatter(dates[lstm_es_breaches], returns[lstm_es_breaches], color="green", marker="x", label="LSTM ES Breach")
    ax_es.scatter(dates[mc_es_breaches], returns[mc_es_breaches], color="orange", marker="x", label="Monte Carlo ES Breach")
    ax_es.axhline(0, color="gray", linestyle=":")

    breach_text = (
        f"Hist ES Breach Rate: {hist_es_breach_rate:.2f}%\n"
        f"LSTM ES Breach Rate: {lstm_es_breach_rate:.2f}%\n"
        f"Monte Carlo ES Breach Rate: {mc_es_breach_rate:.2f}%"
    )
    ax_es.text(0.02, 0.98, breach_text, transform=ax_es.transAxes,
               verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax_es.set_title(f"{portfolio_id}: Expected Shortfall Measures (with Breaches)")
    ax_es.set_xlabel("Date")
    ax_es.set_ylabel("Expected Shortfall / Returns")
    ax_es.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax_es.grid(True)

    # Save plot
    save_path = os.path.join(plots_dir, f"{portfolio_id}_risk_comparison_with_breaches.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path  

def plot_stress_period_analysis_covid(results_df, stress_stats, portfolio_name, plots_dir):
    """
    Create detailed visualizations comparing Historical, LSTM, and Monte Carlo VaR during stress period
    with improved legend placement.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)

    # 1. Returns and Risk Measures Plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results_df['Date'], results_df['Returns'], label='Portfolio Returns', color='blue', alpha=0.7)
    ax1.plot(results_df['Date'], results_df['Historical_VaR'], label='Historical VaR (95%)', color='red', linestyle='--')
    ax1.plot(results_df['Date'], results_df['LSTM_VaR'], label='LSTM VaR (95%)', color='green', linestyle='--')
    ax1.plot(results_df['Date'], results_df['MonteCarlo_VaR'], label='Monte Carlo VaR (95%)', color='orange', linestyle='--')
    
    # Highlight stress period
    stress_mask = results_df['Is_Stress_Period']
    stress_dates = results_df[stress_mask]['Date']
    ax1.axvspan(stress_dates.iloc[0], stress_dates.iloc[-1],
                color='gray', alpha=0.2, label='COVID-19 Stress Period')

    # Plot breaches
    hist_breach_dates = results_df[results_df['Historical_VaR_Breach']]['Date']
    hist_breach_returns = results_df[results_df['Historical_VaR_Breach']]['Returns']
    lstm_breach_dates = results_df[results_df['LSTM_VaR_Breach']]['Date']
    lstm_breach_returns = results_df[results_df['LSTM_VaR_Breach']]['Returns']
    mc_breach_dates = results_df[results_df['MonteCarlo_VaR_Breach']]['Date']
    mc_breach_returns = results_df[results_df['MonteCarlo_VaR_Breach']]['Returns']
    ax1.scatter(hist_breach_dates, hist_breach_returns, color='red', marker='x',
                label='Historical VaR Breaches', zorder=5)
    ax1.scatter(lstm_breach_dates, lstm_breach_returns, color='green', marker='x',
                label='LSTM VaR Breaches', zorder=5)
    ax1.scatter(mc_breach_dates, mc_breach_returns, color='orange', marker='x',
                label='Monte Carlo VaR Breaches', zorder=5)

    ax1.set_title(f'{portfolio_name}: Returns and VaR Measures During COVID-19 Stress Period')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Return/Risk Level')
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=10)
    ax1.grid(True)

    # Add stress period statistics
    stats_text = (
        f"Stress Period Statistics:\n"
        f"Mean Return: {stress_stats['Stress_Period_Returns_Mean']:.4%}\n"
        f"Worst Loss: {stress_stats['Stress_Period_Worst_Loss']:.4%}\n"
        f"Historical VaR Breach Rate: {stress_stats['Historical_VaR_Breach_Rate']:.1f}%\n"
        f"LSTM VaR Breach Rate: {stress_stats['LSTM_VaR_Breach_Rate']:.1f}%\n"
        f"Monte Carlo VaR Breach Rate: {stress_stats['MonteCarlo_VaR_Breach_Rate']:.1f}%\n"
        f"Historical VaR Change: {stress_stats['Historical_VaR_Change_Pct']:.1f}%\n"
        f"LSTM VaR Change: {stress_stats['LSTM_VaR_Change_Pct']:.1f}%\n"
        f"Monte Carlo VaR Change: {stress_stats['MonteCarlo_VaR_Change_Pct']:.1f}%"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. VaR Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results_df['Date'], results_df['Historical_VaR'], label='Historical VaR', color='red')
    ax2.plot(results_df['Date'], results_df['LSTM_VaR'], label='LSTM VaR', color='green')
    ax2.plot(results_df['Date'], results_df['MonteCarlo_VaR'], label='Monte Carlo VaR', color='orange')
    ax2.axvspan(stress_dates.iloc[0], stress_dates.iloc[-1],
                color='gray', alpha=0.2, label='COVID-19 Stress Period')
    ax2.set_title('Historical vs LSTM vs Monte Carlo VaR Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('VaR')
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True)

    # 3. Breach Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    # Calculate cumulative breaches
    hist_cum_breaches = np.cumsum(results_df['Historical_VaR_Breach'])
    lstm_cum_breaches = np.cumsum(results_df['LSTM_VaR_Breach'])
    mc_cum_breaches = np.cumsum(results_df['MonteCarlo_VaR_Breach'])
    ax3.plot(results_df['Date'], hist_cum_breaches, label='Cumulative Historical VaR Breaches', color='red')
    ax3.plot(results_df['Date'], lstm_cum_breaches, label='Cumulative LSTM VaR Breaches', color='green')
    ax3.plot(results_df['Date'], mc_cum_breaches, label='Cumulative Monte Carlo VaR Breaches', color='orange')
    ax3.axvspan(stress_dates.iloc[0], stress_dates.iloc[-1],
                color='gray', alpha=0.2, label='COVID-19 Stress Period')
    ax3.set_title('Cumulative VaR Breaches')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Number of Breaches')
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    ax3.grid(True)

    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(plots_dir, f"{portfolio_name}_covid_stress_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path

def plot_return_distributions(portfolio_id, predicted_returns, actual_returns, plots_dir):
    """
    Modified distribution plots with better legend placement
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Histogram of predicted returns with normal fit
    ax1 = fig.add_subplot(gs[0, 0])
    mu_pred = np.mean(predicted_returns)
    sigma_pred = np.std(predicted_returns)
    n_pred, bins_pred, _ = ax1.hist(predicted_returns, bins=50, density=True, 
                                   alpha=0.7, color='blue', label='Predicted Returns')
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu_pred, sigma_pred)
    ax1.plot(x, p, 'k--', linewidth=2, label='Normal Fit')
    ax1.set_title(f'{portfolio_id}: Distribution of Predicted Returns')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99),
            ncol=2,  # Arrange items in two columns to save vertical space
            fancybox=True, shadow=True,
            bbox_transform=ax1.transAxes)
    ax1.grid(True)
    
    # Stats box in upper left
    stats_text = f'Mean: {mu_pred:.6f}\nStd: {sigma_pred:.6f}\n'
    stats_text += f'Skewness: {stats.skew(predicted_returns):.3f}\n'
    stats_text += f'Kurtosis: {stats.kurtosis(predicted_returns):.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Q-Q plot for predicted returns
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(predicted_returns, dist="norm", plot=ax2)
    ax2.set_title(f'{portfolio_id}: Q-Q Plot of Predicted Returns')
    
    # 3. Histogram of actual returns with normal fit
    ax3 = fig.add_subplot(gs[1, 0])
    mu_act = np.mean(actual_returns)
    sigma_act = np.std(actual_returns)
    n_act, bins_act, _ = ax3.hist(actual_returns, bins=50, density=True, 
                                 alpha=0.7, color='green', label='Actual Returns')
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu_act, sigma_act)
    ax3.plot(x, p, 'k--', linewidth=2, label='Normal Fit')
    ax3.set_title(f'{portfolio_id}: Distribution of Actual Returns')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Density')
    ax3.legend(loc='upper right')  # Move legend to upper right
    ax3.grid(True)
    
    # Stats box in upper left
    stats_text = f'Mean: {mu_act:.6f}\nStd: {sigma_act:.6f}\n'
    stats_text += f'Skewness: {stats.skew(actual_returns):.3f}\n'
    stats_text += f'Kurtosis: {stats.kurtosis(actual_returns):.3f}'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Q-Q plot for actual returns
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(actual_returns, dist="norm", plot=ax4)
    ax4.set_title(f'{portfolio_id}: Q-Q Plot of Actual Returns')
    
    # 5. Distribution comparison
    ax5 = fig.add_subplot(gs[:, 2])
    
    # Create common bins for both distributions
    min_val = min(min(predicted_returns), min(actual_returns))
    max_val = max(max(predicted_returns), max(actual_returns))
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
    ax5.legend(loc='upper right')  # Move legend to upper right
    ax5.grid(True)
    
    # Move test statistics to upper left
    _, p_value_pred = stats.normaltest(predicted_returns)
    _, p_value_act = stats.normaltest(actual_returns)
    test_text = (f'Normality Test p-values:\n'
                f'Predicted: {p_value_pred:.6f}\n'
                f'Actual: {p_value_act:.6f}\n\n'
                f'Std Dev Comparison:\n'
                f'Predicted: {sigma_pred:.6f}\n'
                f'Actual: {sigma_act:.6f}')
    ax5.text(0.02, 0.98, test_text, transform=ax5.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"portfolio_{portfolio_id}_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_lstm_analysis(model, X_test, actual_returns, dates, portfolio_name, plots_dir):
    """
    LSTM analysis plots with predicted volatility only.
    """
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)

    # 1. Temporal Sensitivity Analysis
    ax1 = fig.add_subplot(gs[0, :])
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        base_mean, base_std = model(X_tensor)
        base_mean = base_mean.cpu().numpy()
        base_std = base_std.cpu().numpy()
        
        # Calculate sensitivity up to 6 months
        sensitivities = []
        window_sizes = [5, 21, 63, 126]  # 1 week, 1 month, 3 months, 6 months
        window_labels = ['1 week', '1 month', '3 months', '6 months']
        
        for window in window_sizes:
            window_sensitivity = []
            for i in range(len(X_test)):
                perturbed = X_test[i].copy()
                perturbed[-window:] *= 1.01  # 1% shock to recent data
                
                perturbed_tensor = torch.tensor(perturbed.reshape(1, -1, 1), 
                                              dtype=torch.float32).to(device)
                shock_mean, shock_std = model(perturbed_tensor)
                
                mean_change = float(abs((shock_mean.cpu().numpy() - base_mean[i]) / base_mean[i]))
                std_change = float(abs((shock_std.cpu().numpy() - base_std[i]) / base_std[i]))
                window_sensitivity.append(mean_change + std_change)
            
            sensitivities.append(np.mean(window_sensitivity) * 100)
    
    # Plot bar chart
    bars = ax1.bar(window_labels, sensitivities, alpha=0.7)
    ax1.set_title('LSTM Sensitivity to Historical Data Windows')
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('Average Prediction Change (%)')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    ax1.grid(True, alpha=0.3)

    # 2. Volatility Analysis - Predicted only
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Calculate predicted volatility
    predicted_vol = (base_std.squeeze() 
                    * np.sqrt(252)  # Annualization factor
                    * 100)  # Convert to percentage
    
    valid_dates = dates[-len(predicted_vol):]
    
    ax2.plot(valid_dates, predicted_vol, 
             label='LSTM Predicted Volatility', alpha=0.7, color='blue')
    ax2.set_title('LSTM Predicted Volatility (Annualized %)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Annualized Volatility (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Prediction Error Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    pred_error = base_mean.squeeze() - actual_returns[-len(base_mean):]
    
    ax3.scatter(actual_returns[-len(pred_error):], pred_error, 
                alpha=0.5, label='Prediction Errors')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('LSTM Prediction Errors vs Returns')
    ax3.set_xlabel('Actual Returns')
    ax3.set_ylabel('Prediction Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Market Regime Analysis
    ax4 = fig.add_subplot(gs[2, :])
    valid_rolling_vol = pd.Series(actual_returns).rolling(21).std()[-len(pred_error):]
    vol_regimes = pd.qcut(valid_rolling_vol, q=3, labels=['Low', 'Medium', 'High'])
    regime_errors = {}
    
    for regime in ['Low', 'Medium', 'High']:
        mask = vol_regimes == regime
        regime_errors[regime] = np.mean(np.abs(pred_error[mask]))
    
    ax4.bar(regime_errors.keys(), regime_errors.values(), alpha=0.7)
    ax4.set_title('LSTM Prediction Error by Volatility Regime')
    ax4.set_xlabel('Volatility Regime')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{portfolio_name}_lstm_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path