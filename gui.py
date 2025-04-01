import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import data
import portfolio

class ETFOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EuroETF Optimizer")
        self.prices = data.load_etf_data()
        self.tickers = self.prices.columns
        
        # Labels and inputs
        tk.Label(root, text="Portfolio Weights").pack()
        self.weight_vars = {}
        for ticker in self.tickers:
            frame = ttk.Frame(root)
            frame.pack(fill='x', padx=5, pady=2)
            tk.Label(frame, text=f"{ticker} (%)", width=10).pack(side='left')
            var = tk.DoubleVar(value=20.0)  # Default 20%
            self.weight_vars[ticker] = var
            tk.Entry(frame, textvariable=var, width=10).pack(side='left')
        
        # Buttons
        tk.Button(root, text="Optimize Automatically", command=self.optimize).pack(pady=5)
        tk.Button(root, text="Run Backtest", command=self.run_backtest).pack(pady=5)
        
        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    def optimize(self):
        returns = portfolio.calculate_returns(self.prices)
        weights = portfolio.optimize_portfolio(returns)
        for i, ticker in enumerate(self.tickers):
            self.weight_vars[ticker].set(weights[i] * 100)
        self.plot_weights(weights)

    def run_backtest(self):
        weights = np.array([self.weight_vars[t].get() / 100 for t in self.tickers])
        if not 0.99 <= weights.sum() <= 1.01:
            tk.messagebox.showerror("Error", "Weights must sum to 100%")
            return
        cumulative_return, _, _, _ = portfolio.backtest_portfolio(self.prices, weights, self.tickers)
        self.plot_performance(cumulative_return)

    def plot_weights(self, weights):
        self.ax.clear()
        self.ax.pie(weights, labels=self.tickers, autopct='%1.1f%%', startangle=90)
        self.ax.set_title("Portfolio Weights")
        self.canvas.draw()

    def plot_performance(self, cumulative_return):
        self.ax.clear()
        self.ax.plot(cumulative_return.index, cumulative_return, label='Portfolio')
        self.ax.set_title("Backtested Performance")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Cumulative Return")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ETFOptimizerGUI(root)
    root.mainloop()