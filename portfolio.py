import pandas as pd
import numpy as np
from scipy.optimize import minimize
import data  # Assumed module for data loading
import sentiment
from pmdarima import auto_arima

# Load ETF prices (assumed to be provided by data.py)
prices = data.load_etf_data()

def calculate_returns(prices, start_date=None, end_date=None, use_sentiment=False):
    """
    Calculate daily returns from prices, optionally adjusting with sentiment.
    """
    if start_date and end_date:
        prices = prices.loc[start_date:end_date]
    returns = prices.pct_change().dropna()
    if use_sentiment:
        sentiment_scores = sentiment.fetch_rss_sentiment(prices)
        returns = sentiment.adjust_returns(returns, sentiment_scores)
    return returns

def portfolio_performance(weights, returns, risk_free_rate=0.01):
    """
    Compute portfolio expected return, volatility, and Sharpe Ratio.
    """
    exp_return = np.sum(weights * returns.mean()) * 252  # Annualized return
    portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = (exp_return - risk_free_rate) / volatility if volatility != 0 else 0
    return exp_return, volatility, sharpe_ratio

def optimize_portfolio_sharpe(returns, risk_free_rate=0.01):
    """
    Optimize portfolio weights to maximize the Sharpe Ratio using Mean-Variance Optimization.
    """
    num_assets = returns.shape[1]
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    )
    bounds = tuple((0.05, 0.4) for _ in range(num_assets))  # Weights between 5% and 40%
    
    def objective(weights):
        _, _, sharpe = portfolio_performance(weights, returns, risk_free_rate)
        return -sharpe  # Minimize negative Sharpe Ratio to maximize it
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else initial_weights

def equal_weights(num_assets):
    """
    Return equal weights for the given number of assets.
    """
    return np.array([1.0 / num_assets] * num_assets)

def compute_efficient_frontier(returns, num_portfolios=20):
    """
    Compute the Efficient Frontier by minimizing volatility for various target returns.
    """
    min_return = returns.mean().min() * 252
    max_return = returns.mean().max() * 252
    target_returns = np.linspace(min_return, max_return, num_portfolios)
    efficient_portfolios = []
    
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * returns.mean()) * 252 - target}
        )
        bounds = tuple((0.05, 0.4) for _ in range(returns.shape[1]))
        initial_weights = np.array([1.0 / returns.shape[1]] * returns.shape[1])
        result = minimize(
            lambda w: portfolio_performance(w, returns)[1],  # Minimize volatility
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            efficient_portfolios.append({
                'return': target,
                'volatility': result.fun,
                'weights': result.x
            })
    return efficient_portfolios

def forecast_returns(returns, horizon=30, end_date=None):
    """
    Forecast future returns for each ETF using ARIMA, optionally up to a specified end date.
    
    Parameters:
    - returns: DataFrame of historical returns.
    - horizon: Number of periods to forecast.
    - end_date: Optional end date for historical data to use in forecasting (e.g., today).
    
    Returns:
    - DataFrame of forecasted returns.
    """
    if end_date:
        returns = returns.loc[:end_date]
    
    forecasts = {}
    for etf in returns.columns:
        model = auto_arima(returns[etf], seasonal=False, suppress_warnings=True)
        forecast = model.predict(n_periods=horizon)
        forecasts[etf] = forecast
    
    forecast_index = pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')
    forecast_df = pd.DataFrame(forecasts, index=forecast_index)
    return forecast_df