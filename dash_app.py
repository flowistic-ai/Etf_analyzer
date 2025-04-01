import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os  # For file operations
import data  # Assumes data.py contains fetch_etf_data and load_etf_data
import portfolio  # Assumes portfolio.py contains calculate_returns, optimize_portfolio_sharpe, etc.
import backtest  # Assumes backtest.py contains backtest_portfolio
import statsmodels.api as sm
from datetime import datetime, timedelta
import traceback

# Error handling wrapper for callbacks
def handle_callback_error(func):
    """Wrap callbacks to handle errors gracefully, ignoring PreventUpdate."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PreventUpdate:
            raise  # Pass PreventUpdate silently
        except Exception as e:
            print(f"Callback error: {str(e)}")
            print(traceback.format_exc())
            return dash.no_update
    return wrapper

# Initialize Dash app with error handling
try:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    
    # Load ETF data with validation
    prices_full = data.load_etf_data()
    if prices_full is None or prices_full.empty:
        raise ValueError("Failed to load ETF data")
    
    # Load DAX data with proper fetching
    try:
        dax_prices = data.get_dax_data()
        if dax_prices is None or dax_prices.empty:
            print("Warning: Failed to load DAX data")
    except Exception as e:
        print(f"Error fetching DAX data: {e}")
        dax_prices = pd.Series()
    
    # Load DAX data with proper fetching
    dax_ticker = "^GDAXI"
    dax_data = data.fetch_etf_data(dax_ticker)
    dax_prices = dax_data['Close'] if dax_data is not None and not dax_data.empty else pd.Series()
    if dax_prices.empty:
        print("Warning: Failed to load DAX data")
    
    tickers = prices_full.columns
    
    # Define today's date
    today = pd.Timestamp.now().normalize()
    
    # Initial parameters with validation
    initial_weights = [20] * len(tickers)
    initial_start_date = max(
        (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        prices_full.index.min().strftime('%Y-%m-%d')
    )
    initial_end_date = min(
        datetime.now().strftime('%Y-%m-%d'),
        prices_full.index.max().strftime('%Y-%m-%d')
    )

    # Layout
    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("EuroETF Optimizer Dashboard", className="text-center text-light mb-4"))),
        dbc.Tabs([
            # Portfolio Analysis Tab
            dbc.Tab(label="Portfolio Analysis", children=[
                dbc.Row([
                    # Control Panel
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Control Panel", className="bg-primary text-white"),
                            dbc.CardBody([
                                html.Label("Weight Strategy", className="text-light"),
                                dcc.Dropdown(
                                    id="strategy-dropdown",
                                    options=[
                                        {'label': 'Manual', 'value': 'Manual'},
                                        {'label': 'Max Sharpe', 'value': 'Maximize Sharpe Ratio'},
                                        {'label': 'Equal', 'value': 'Equal Weighting'}
                                    ],
                                    value='Manual',
                                    className="mb-3"
                                ),
                                html.Div(id="weights-section", children=[
                                    html.Label("Portfolio Weights (%)", className="text-light"),
                                    *[dbc.Row([
                                        dbc.Col(html.Label(f"{etf}", className="text-light"), width=4),
                                        dbc.Col(dcc.Slider(
                                            id=f"weight-{etf.replace('.', '_')}",
                                            min=5, max=40, step=0.1, value=20,
                                            marks={5: '5%', 20: '20%', 40: '40%'},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ), width=8)
                                    ], className="mb-2") for etf in tickers],
                                    html.Div(id="weights-sum", className="text-warning")
                                ]),
                                html.Label("Date Range", className="text-light"),
                                dcc.DatePickerRange(
                                    id='date-picker',
                                    min_date_allowed=prices_full.index.min(),
                                    max_date_allowed=prices_full.index.max(),
                                    start_date=initial_start_date,
                                    end_date=initial_end_date,
                                    display_format='YYYY-MM-DD',
                                    className="mb-3"
                                ),
                                html.P("Note: Dates beyond today include fictional data.", className="text-warning mt-2"),
                                html.Label("Return Type", className="text-light"),
                                dcc.RadioItems(
                                    id="return-type",
                                    options=[{'label': 'Historical', 'value': 'historical'}, {'label': 'Predicted', 'value': 'predicted'}],
                                    value='historical',
                                    labelClassName="text-light",
                                    className="mb-3"
                                ),
                                dbc.Button("Run Analysis", id="run-backtest", n_clicks=0, color="success", className="w-100"),
                                html.Div(id="status-message", className="text-success mt-2")
                            ])
                        ], className="bg-dark p-3")
                    ], width=4),
                    # Outputs
                    dbc.Col([
                        dcc.Loading(
                            id="loading-backtest",
                            type="default",
                            children=dcc.Graph(id="backtest-plot")
                        ),
                        dcc.Loading(
                            id="loading-efficient-frontier",
                            type="default",
                            children=dcc.Graph(id="efficient-frontier-plot")
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Loading(
                                id="loading-daily-returns",
                                type="default",
                                children=dcc.Graph(id="daily-returns-plot")
                            ), width=6),
                            dbc.Col(dcc.Loading(
                                id="loading-weights-pie",
                                type="default",
                                children=dcc.Graph(id="weights-pie")
                            ), width=6)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Key Metrics", className="bg-primary text-white"),
                            dbc.CardBody(id="metrics-cards", className="bg-dark")
                        ])
                    ], width=8)
                ])
            ]),
            # Predictions Tab
            dbc.Tab(label="Predictions", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Generate Forecasts", id="generate-forecasts", n_clicks=0, color="primary", className="mb-3"),
                        dcc.Loading(
                            id="loading-forecast",
                            type="default",
                            children=dcc.Graph(id="forecast-plot")
                        )
                    ])
                ])
            ]),
            # Stress Tests Tab
            dbc.Tab(label="Stress Tests", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Stress Scenario", className="text-light"),
                        dcc.Dropdown(
                            id="stress-scenario",
                            options=[
                                {'label': '2008 Financial Crisis', 'value': '2008'},
                                {'label': 'COVID-19 Crash', 'value': '2020'}
                            ],
                            value='2008',
                            className="mb-3"
                        ),
                        dbc.Button("Run Stress Test", id="run-stress-test", n_clicks=0, color="danger", className="mb-3"),
                        dcc.Loading(
                            id="loading-stress",
                            type="default",
                            children=dcc.Graph(id="stress-plot")
                        )
                    ])
                ])
            ]),
            # News & Sentiment Tab
            dbc.Tab(label="News & Sentiment", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Latest News", className="bg-primary text-white"),
                            dbc.CardBody(id="news-feed", className="bg-dark")
                        ])
                    ])
                ])
            ]),
            # Summary Tab
            dbc.Tab(label="Summary", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Analysis Summary", className="text-light"),
                        html.Div(id="summary-text", className="text-light")
                    ])
                ])
            ])
        ]),
        dcc.Store(id="summary-store")
    ], fluid=True, className="bg-dark")

except Exception as e:
    print(f"Error initializing app: {str(e)}")
    print(traceback.format_exc())
    raise

# Helper function to generate summary
def generate_summary(metrics):
    sharpe = metrics['sharpe']
    interp = "good" if sharpe > 1 else "moderate" if sharpe > 0.5 else "poor"
    return (
        f"Your portfolio has a Sharpe Ratio of {sharpe:.2f}, indicating {interp} risk-adjusted performance. "
        f"The expected annual return is {metrics['return']:.2%}, with an annual volatility of {metrics['volatility']:.2%}."
    )

# Callbacks with error handling

# Toggle sliders based on strategy
@callback(
    [Output(f"weight-{etf.replace('.', '_')}", "disabled") for etf in tickers],
    Input("strategy-dropdown", "value")
)
@handle_callback_error
def toggle_sliders(strategy):
    return [strategy != "Manual"] * len(tickers)

# Update weights based on strategy
@callback(
    [Output(f"weight-{etf.replace('.', '_')}", "value") for etf in tickers],
    [Input("strategy-dropdown", "value"), Input("date-picker", "start_date"), Input("date-picker", "end_date")],
    prevent_initial_call=True
)
@handle_callback_error
def update_weights(strategy, start_date, end_date):
    if not start_date or not end_date:
        raise PreventUpdate
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    if strategy == "Manual":
        return [dash.no_update] * len(tickers)
    prices = prices_full.loc[start_date:end_date]
    if prices.empty:
        raise ValueError("No price data available for the selected date range")
    returns = portfolio.calculate_returns(prices)
    if strategy == "Maximize Sharpe Ratio":
        weights = portfolio.optimize_portfolio_sharpe(returns)
    elif strategy == "Equal Weighting":
        weights = portfolio.equal_weights(len(tickers))
    return [w * 100 for w in weights]

# Update weights sum
@callback(
    Output("weights-sum", "children"),
    [Input(f"weight-{etf.replace('.', '_')}", "value") for etf in tickers]
)
@handle_callback_error
def update_weights_sum(*weights):
    total = sum(weights)
    color = "text-danger" if not 99 <= total <= 101 else "text-success"
    return html.Span(f"Total: {total:.1f}%", className=color)

# Run backtest and update outputs
@callback(
    [Output("backtest-plot", "figure"), Output("daily-returns-plot", "figure"),
     Output("metrics-cards", "children"), Output("weights-pie", "figure"),
     Output("status-message", "children"), Output("efficient-frontier-plot", "figure"),
     Output("summary-store", "data")],
    Input("run-backtest", "n_clicks"),
    [State(f"weight-{etf.replace('.', '_')}", "value") for etf in tickers] +
    [State("date-picker", "start_date"), State("date-picker", "end_date"),
     State("return-type", "value")]
)
@handle_callback_error
def run_backtest(n_clicks, *args):
    if n_clicks == 0:
        raise PreventUpdate

    weights_input = args[:len(tickers)]
    start_date, end_date, return_type = args[len(tickers):]
    weights = np.array(weights_input) / 100

    if not 0.99 <= sum(weights) <= 1.01:
        return [dash.no_update] * 6 + ["Weights must sum to ~100%"]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    prices = prices_full.loc[start_date:end_date]
    if prices.empty:
        raise ValueError("No price data available for the selected date range")
    returns = portfolio.calculate_returns(prices, use_sentiment=(return_type == "predicted"))

    # Check for fictional data in the date range
    status_message = "Analysis completed."
    if end_date > today:
        status_message += " Warning: Includes fictional future data."

    # Backtest
    cumulative_return, ann_return, ann_vol, max_dd = backtest.backtest_portfolio(prices, weights, tickers, returns)
    portfolio_returns = returns.dot(weights)
    dax_returns = dax_prices.loc[start_date:end_date].pct_change().dropna()
    dax_cumulative = (1 + dax_returns).cumprod()

    # Alpha and Beta
    common_dates = portfolio_returns.index.intersection(dax_returns.index)
    if common_dates.empty:
        alpha, beta = 0, 0
    else:
        X = sm.add_constant(dax_returns.loc[common_dates])
        model = sm.OLS(portfolio_returns.loc[common_dates], X).fit()
        alpha = model.params.iloc[0] * 252
        beta = model.params.iloc[1]

    # Plots
    backtest_fig = go.Figure()
    backtest_fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return, name="Portfolio", line=dict(color="#00ff00")))
    backtest_fig.add_trace(go.Scatter(x=dax_cumulative.index, y=dax_cumulative, name="DAX", line=dict(color="#00ffff")))
    if end_date > today:
        backtest_fig.add_vline(x=today, line_dash="dash", line_color="red", annotation_text="Start of fictional data", annotation_position="top left")
    backtest_fig.update_layout(title="Portfolio vs DAX", xaxis_title="Date", yaxis_title="Cumulative Return", template="plotly_dark")

    daily_fig = go.Figure()
    daily_fig.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns, name="Portfolio", line=dict(color="#00ff00")))
    daily_fig.add_trace(go.Scatter(x=dax_returns.index, y=dax_returns, name="DAX", line=dict(color="#00ffff")))
    daily_fig.update_layout(title="Daily Returns", xaxis_title="Date", yaxis_title="Return", template="plotly_dark")

    pie_fig = go.Figure(go.Pie(labels=tickers, values=weights, textinfo='label+percent', hole=0.3))
    pie_fig.update_layout(title="Portfolio Allocation", template="plotly_dark")

    # Efficient Frontier
    ef_portfolios = portfolio.compute_efficient_frontier(returns)
    ef_fig = go.Figure()
    ef_fig.add_trace(go.Scatter(x=[p['volatility'] for p in ef_portfolios], y=[p['return'] for p in ef_portfolios], mode='lines', name='Efficient Frontier'))
    curr_vol, curr_retn, _ = portfolio.portfolio_performance(weights, returns)
    ef_fig.add_trace(go.Scatter(x=[curr_vol], y=[curr_retn], mode='markers', name='Current Portfolio', marker=dict(color='red', size=10)))
    ef_fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return", template="plotly_dark")

    # Metrics
    metrics_cards = dbc.Row([
        dbc.Col(dbc.Card([html.H5("Cumulative Return"), html.H3(f"{(cumulative_return.iloc[-1] - 1):.2%}")], className="bg-success text-center p-2"), width=3),
        dbc.Col(dbc.Card([html.H5("Alpha"), html.H3(f"{alpha:.2%}")], className="bg-info text-center p-2"), width=3),
        dbc.Col(dbc.Card([html.H5("Volatility"), html.H3(f"{ann_vol:.2%}")], className="bg-warning text-center p-2"), width=3),
        dbc.Col(dbc.Card([html.H5("Max Drawdown"), html.H3(f"{max_dd:.2%}")], className="bg-danger text-center p-2"), width=3)
    ])

    # Summary
    metrics = {'return': ann_return, 'volatility': ann_vol, 'sharpe': (ann_return - 0.01) / ann_vol if ann_vol != 0 else 0}
    summary = generate_summary(metrics)

    return backtest_fig, daily_fig, metrics_cards, pie_fig, status_message, ef_fig, summary

# Display summary
@callback(
    Output("summary-text", "children"),
    Input("summary-store", "data")
)
@handle_callback_error
def display_summary(data):
    return data or "Run analysis to see summary."

# Generate forecasts
@callback(
    Output("forecast-plot", "figure"),
    Input("generate-forecasts", "n_clicks"),
    [State(f"weight-{etf.replace('.', '_')}", "value") for etf in tickers]
)
@handle_callback_error
def generate_forecasts(n_clicks, *weights_input):
    if n_clicks == 0:
        raise PreventUpdate
    weights = np.array(weights_input) / 100
    if not 0.99 <= sum(weights) <= 1.01:
        return go.Figure().update_layout(title="Weights must sum to 100%", template="plotly_dark")
    
    # Use only historical data up to today for forecasting
    hist_prices = prices_full.loc[:today]
    if hist_prices.empty:
        raise ValueError("No historical price data available up to today")
    hist_returns = portfolio.calculate_returns(hist_prices)
    forecast_returns = portfolio.forecast_returns(hist_returns)
    hist_port_returns = hist_returns.dot(weights)
    forecast_port_returns = forecast_returns.dot(weights)
    cum_hist = (1 + hist_port_returns).cumprod()
    cum_forecast = cum_hist.iloc[-1] * (1 + forecast_port_returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_hist.index, y=cum_hist, name="Historical", line=dict(color="#00ff00")))
    fig.add_trace(go.Scatter(x=cum_forecast.index, y=cum_forecast, name="Forecast", line=dict(color="#ffa500", dash="dash")))
    fig.add_vline(x=today, line_dash="dash", line_color="red", annotation_text="Today", annotation_position="top left")
    fig.update_layout(title="Portfolio Cumulative Returns with Forecast", xaxis_title="Date", yaxis_title="Cumulative Return", template="plotly_dark")
    return fig

# Run stress test
@callback(
    Output("stress-plot", "figure"),
    Input("run-stress-test", "n_clicks"),
    [State("stress-scenario", "value")] + [State(f"weight-{etf.replace('.', '_')}", "value") for etf in tickers]
)
@handle_callback_error
def run_stress_test(n_clicks, scenario, *weights_input):
    if n_clicks == 0:
        raise PreventUpdate
    weights = np.array(weights_input) / 100
    if not 0.99 <= sum(weights) <= 1.01:
        return go.Figure().update_layout(title="Weights must sum to 100%", template="plotly_dark")
    dates = {'2008': ("2008-09-01", "2009-03-01"), '2020': ("2020-02-20", "2020-03-23")}
    start_date, end_date = dates.get(scenario, ("2008-09-01", "2009-03-01"))
    prices_stress = prices_full.loc[start_date:end_date]
    if prices_stress.empty:
        raise ValueError(f"No price data available for stress test period {start_date} to {end_date}")
    returns_stress = portfolio.calculate_returns(prices_stress)
    port_returns_stress = returns_stress.dot(weights)
    cum_stress = (1 + port_returns_stress).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_stress.index, y=cum_stress, name="Portfolio", line=dict(color="#ff0000")))
    fig.update_layout(title=f"Portfolio during {scenario} Stress", xaxis_title="Date", yaxis_title="Cumulative Return", template="plotly_dark")
    return fig

# Update news feed (placeholder)
@callback(
    Output("news-feed", "children"),
    Input("date-picker", "end_date")
)
@handle_callback_error
def update_news(end_date):
    if not end_date:
        raise PreventUpdate
    news = [{"title": "Sample News", "date": end_date, "sentiment": 0.7}]  # Placeholder
    return [dbc.Card([
        html.H5(item["title"]), html.P(f"Date: {item['date']} | Sentiment: {item['sentiment']:.2f}")
    ], className="mb-2 bg-secondary") for item in news]

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error running app: {str(e)}")
        print(traceback.format_exc())