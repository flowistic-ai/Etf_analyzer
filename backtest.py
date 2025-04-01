import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
import portfolio

def backtest_portfolio(prices, weights, tickers, returns, risk_free_rate=0.01):
    """
    Run a backtest on the portfolio, calculating key performance metrics.
    
    Parameters:
    - prices: DataFrame of ETF prices over the backtest period.
    - weights: Array of portfolio weights for each ETF.
    - tickers: List of ETF tickers.
    - returns: DataFrame of daily returns for each ETF.
    - risk_free_rate: Risk-free rate for Sharpe and Sortino ratios (default: 0.01).
    
    Returns:
    - cumulative_return: Series of cumulative portfolio returns.
    - annualized_return: Annualized return of the portfolio.
    - annualized_volatility: Annualized volatility of the portfolio.
    - max_drawdown: Maximum drawdown during the backtest period.
    - sharpe_ratio: Sharpe ratio of the portfolio.
    - sortino_ratio: Sortino ratio of the portfolio.
    """
    if returns.shape[1] != len(weights):
        raise ValueError(f"Shape mismatch: returns has {returns.shape[1]} columns, weights has {len(weights)} elements")
    
    # Calculate portfolio daily returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate cumulative returns
    cumulative_return = (1 + portfolio_returns).cumprod()
    
    # Calculate annualized return (assuming 252 trading days per year)
    annualized_return = (cumulative_return.iloc[-1] ** (252 / len(cumulative_return))) - 1
    
    # Calculate annualized volatility
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Calculate maximum drawdown
    rolling_max = cumulative_return.cummax()
    drawdowns = (cumulative_return - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate Sharpe ratio
    excess_returns = portfolio_returns - risk_free_rate / 252
    sharpe_ratio = (excess_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() != 0 else 0
    
    # Calculate Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
    
    return cumulative_return, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio

def stress_test_portfolio(returns, weights, scenario='crash', crash_pct=0.20, vol_multiplier=2):
    """
    Simulate stress scenarios on the portfolio.
    
    Parameters:
    - returns: DataFrame of daily returns for each ETF.
    - weights: Array of portfolio weights for each ETF.
    - scenario: Type of stress test ('crash' or 'high_vol').
    - crash_pct: Percentage drop for the crash scenario (default: 20%).
    - vol_multiplier: Multiplier for the high volatility scenario (default: 2x).
    
    Returns:
    - stress_cumulative: Cumulative returns under the stress scenario.
    """
    if scenario == 'crash':
        stress_returns = returns.copy()
        stress_returns.iloc[0] -= crash_pct  # Apply crash to first day
    elif scenario == 'high_vol':
        stress_returns = returns * vol_multiplier
    else:
        raise ValueError("Invalid stress scenario. Choose 'crash' or 'high_vol'.")
    
    stress_portfolio_returns = stress_returns.dot(weights)
    stress_cumulative = (1 + stress_portfolio_returns).cumprod()
    return stress_cumulative

def plot_backtest(cumulative_return, stress_cumulative=None, scenario=None):
    """
    Plot the backtest results with optional stress scenarios.
    
    Parameters:
    - cumulative_return: Series of cumulative portfolio returns.
    - stress_cumulative: Cumulative returns under the stress scenario (optional).
    - scenario: Name of the stress scenario (optional).
    """
    plt.figure(figsize=(12, 8))
    plt.plot(cumulative_return.index, cumulative_return, label='Base Portfolio', color='blue')
    if stress_cumulative is not None and scenario:
        plt.plot(stress_cumulative.index, stress_cumulative, label=f'{scenario} Scenario', color='red', linestyle='--')
    plt.title('Portfolio Backtest')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_plot.png')
    plt.close()
    print("Backtest plot saved as 'backtest_plot.png'")

def display_results(cumulative_return, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio, tickers, weights):
    """
    Display the backtest results.
    
    Parameters:
    - cumulative_return: Series of cumulative portfolio returns.
    - annualized_return: Annualized return of the portfolio.
    - annualized_volatility: Annualized volatility of the portfolio.
    - max_drawdown: Maximum drawdown during the backtest period.
    - sharpe_ratio: Sharpe ratio of the portfolio.
    - sortino_ratio: Sortino ratio of the portfolio.
    - tickers: List of ETF tickers.
    - weights: Array of portfolio weights for each ETF.
    """
    print("\nBacktested Portfolio Results:")
    print("Weights:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"\nBase Scenario:")
    print(f"Total Cumulative Return: {(cumulative_return.iloc[-1] - 1):.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annual Volatility: {annualized_volatility:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")

def get_user_weights(tickers):
    """
    Prompt the user to enter manual weights for the ETFs.
    
    Parameters:
    - tickers: List of ETF tickers.
    
    Returns:
    - weights: Array of user-input weights.
    """
    weights = []
    total = 0
    print("\nEnter weights as percentages (e.g., 20 for 20%). Total must equal 100%:")
    for ticker in tickers:
        while True:
            try:
                weight = float(input(f"Weight for {ticker} (%): ")) / 100
                if 0.05 <= weight <= 0.4:
                    weights.append(weight)
                    total += weight
                    break
                else:
                    print("Weight must be between 5% and 40%.")
            except ValueError:
                print("Invalid input. Enter a number.")
    if not 0.99 <= total <= 1.01:
        print(f"Total weights ({total:.2%}) must equal 100%. Starting over.")
        return get_user_weights(tickers)
    return np.array(weights)

if __name__ == "__main__":
    # Load ETF data
    prices_full = data.load_etf_data()
    tickers = prices_full.columns
    print("Tickers loaded:", tickers.tolist())
    
    # Prompt user for weight selection
    choice = input("Optimize weights automatically (O) or enter manually (M)? [O/M]: ").upper()
    if choice == 'O':
        returns_full = portfolio.calculate_returns(prices_full)
        weights = portfolio.optimize_portfolio(returns_full)
    elif choice == 'M':
        weights = get_user_weights(tickers)
    else:
        print("Invalid choice. Using optimized weights.")
        returns_full = portfolio.calculate_returns(prices_full)
        weights = portfolio.optimize_portfolio(returns_full)
    
    # Get date range from user
    start_date = input("Enter start date (YYYY-MM-DD, default 2020-01-01): ") or "2020-01-01"
    end_date = input("Enter end date (YYYY-MM-DD, default today): ") or pd.Timestamp.today().strftime("%Y-%m-%d")
    
    # Validate and adjust dates
    latest_date = prices_full.index.max()
    if pd.Timestamp(end_date) > latest_date:
        print(f"End date {end_date} exceeds latest data ({latest_date}). Adjusting to {latest_date}.")
        end_date = latest_date.strftime("%Y-%m-%d")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        print(f"Start date {start_date} after end date {end_date}. Using default range.")
        start_date = "2020-01-01"
    
    # Slice data for the backtest period
    prices = prices_full.loc[start_date:end_date]
    returns = portfolio.calculate_returns(prices)
    
    # Run backtest
    cumulative_return, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio = backtest_portfolio(
        prices, weights, tickers, returns
    )
    
    # Run stress tests
    crash_cumulative = stress_test_portfolio(returns, weights, scenario='crash')
    high_vol_cumulative = stress_test_portfolio(returns, weights, scenario='high_vol')
    
    # Display results
    display_results(cumulative_return, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio, tickers, weights)
    
    # Plot backtest with stress scenarios
    plot_backtest(cumulative_return)
    plot_backtest(cumulative_return, crash_cumulative, "Crash")
    plot_backtest(cumulative_return, high_vol_cumulative, "High Volatility")