import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data  # To load ETF prices
import portfolio  # To use optimization functions

# Load data and calculate returns
prices = data.load_etf_data()
returns = portfolio.calculate_returns(prices)
tickers = prices.columns

# Get optimized weights from portfolio.py
optimal_weights = portfolio.optimize_portfolio(returns)

def plot_performance(prices, weights, tickers):
    """Plot cumulative returns of individual ETFs and the portfolio."""
    # Calculate cumulative returns for each ETF
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate portfolio cumulative returns
    portfolio_returns = returns.dot(weights)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)
    plt.plot(cumulative_returns.index, portfolio_cumulative, label='Optimized Portfolio', linewidth=2, linestyle='--')
    plt.title('Cumulative Returns of ETFs and Optimized Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()
    print("Cumulative returns plot saved as 'cumulative_returns.png'")

def plot_weights(weights, tickers):
    """Plot portfolio weights as a pie chart."""
    plt.figure(figsize=(8, 8))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    plt.title('Optimized Portfolio Weights')
    plt.axis('equal')
    plt.savefig('portfolio_weights.png')
    plt.close()
    print("Weights pie chart saved as 'portfolio_weights.png'")

if __name__ == "__main__":
    # Plot performance and weights
    plot_performance(prices, optimal_weights, tickers)
    plot_weights(optimal_weights, tickers)
    
    # Re-display portfolio metrics for reference
    exp_return, volatility, sharpe = portfolio.portfolio_performance(optimal_weights, returns)
    print("\nPortfolio Metrics:")
    print(f"Expected Annual Return: {exp_return:.2%}")
    print(f"Annual Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")