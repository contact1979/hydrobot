# /dashboard/callbacks.py

import logging
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
from dash.exceptions import PreventUpdate # <--- Correct import location
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import datetime
import pytz

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config # To potentially update config values (use with caution)
from dashboard import data_provider # To fetch data
from trading import trader # To re-enable symbols
# from .. import config
# from . import data_provider
# from ..trading import trader

log = logging.getLogger(__name__)

# --- Register Callbacks ---
# Note: Callbacks need the 'app' instance, which is created in app.py.
# We define functions here and register them in app.py to avoid circular imports.

def register_callbacks(app: Dash):
    """Registers all callbacks for the Dash application."""
    log.info("Registering dashboard callbacks...")

    # --- Interval Update Callback ---
    @app.callback(
        [
            # Overview Tab Outputs
            Output('overview-status-indicators', 'children'),
            Output('overview-total-value', 'children'),
            Output('overview-cash', 'children'),
            Output('overview-open-positions', 'children'),
            Output('overview-pnl-graph', 'figure'),
            Output('overview-trade-table', 'data'),
            Output('overview-trade-table', 'columns'),
            Output('overview-alerts', 'children'),
            # Trading Tab Outputs
            Output('trading-open-positions-table', 'data'),
            Output('trading-open-positions-table', 'columns'),
            Output('trading-history-table', 'data'),
            Output('trading-history-table', 'columns'),
            # Settings Tab Outputs (Update disabled symbols dropdown)
            Output('settings-disabled-symbol-dropdown', 'options'),
            # Market Tab Outputs (Update symbol list dynamically?)
            # Output('market-symbol-dropdown', 'options'), # Optional: Refresh symbol list
        ],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard_on_interval(n):
        """Periodic update triggered by the interval timer."""
        log.debug(f"Dashboard refresh triggered by interval {n}")

        # Fetch data using data_provider functions
        overview_data = data_provider.get_overview_data()
        open_pos_df, history_df = data_provider.get_trading_data()
        perf_df = data_provider.get_performance_data() # For PnL graph

        # --- Process Overview Data ---
        status_indicators = [
            html.P(f"Trading Halted: {'YES' if overview_data.get('halt_status') else 'NO'}",
                   style={'color': 'red' if overview_data.get('halt_status') else 'green'}),
            # TODO: Add status for Model, Data Feeds etc.
        ]
        total_value_str = f"${overview_data.get('total_value', 0):,.2f}"
        cash_str = f"${overview_data.get('cash', 0):,.2f}"
        open_pos_count_str = str(overview_data.get('open_positions_count', 0))
        alerts_children = [html.P("Disabled Symbols: " + ", ".join(overview_data.get('disabled_symbols', [])))]
        # TODO: Add other system alerts

        # PnL Graph
        pnl_fig = go.Figure()
        if not perf_df.empty:
            pnl_fig.add_trace(go.Scatter(x=perf_df.index, y=perf_df['cumulative_pnl'], mode='lines', name='Cumulative PnL'))
            pnl_fig.update_layout(title="Cumulative Realized PnL Over Time", xaxis_title="Time", yaxis_title="PnL (USD)")
        else:
            pnl_fig = create_empty_figure("No PnL data available")

        # Recent Trades Table
        trade_cols = [{"name": i, "id": i} for i in overview_data.get('latest_trades', pd.DataFrame()).columns]
        trade_data = overview_data.get('latest_trades', pd.DataFrame()).to_dict('records')

        # --- Process Trading Data ---
        open_pos_cols = [{"name": i, "id": i} for i in open_pos_df.columns]
        open_pos_data = open_pos_df.to_dict('records')
        history_cols = [{"name": i, "id": i} for i in history_df.columns]
        history_data = history_df.to_dict('records')

        # --- Process Settings Data ---
        disabled_symbols_options = [{'label': s, 'value': s} for s in overview_data.get('disabled_symbols', [])]

        return (
            status_indicators, total_value_str, cash_str, open_pos_count_str, pnl_fig,
            trade_data, trade_cols, alerts_children,
            open_pos_data, open_pos_cols, history_data, history_cols,
            disabled_symbols_options
        )

    # --- Market View Callbacks ---
    @app.callback(
        Output('market-price-chart', 'figure'),
        [Input('market-symbol-dropdown', 'value'),
         Input('market-timeframe-dropdown', 'value')]
    )
    def update_market_chart(selected_symbol, selected_timeframe):
        """Updates the price chart based on selected symbol and timeframe."""
        if not selected_symbol or not selected_timeframe:
            raise PreventUpdate # Or return empty figure

        log.debug(f"Updating market chart for {selected_symbol}, timeframe {selected_timeframe}")
        market_df = data_provider.get_market_data(selected_symbol, selected_timeframe)

        if market_df.empty:
            return create_empty_figure(f"No market data found for {selected_symbol}")

        # Create Candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(x=market_df.index,
                        open=market_df['open'], high=market_df['high'],
                        low=market_df['low'], close=market_df['close'],
                        name='Price'), row=1, col=1)

        # Add Volume bars
        fig.add_trace(go.Bar(x=market_df.index, y=market_df['volume'], name='Volume'), row=2, col=1)

        # Add Technical Indicators (example for SMA)
        if 'sma_fast' in market_df.columns:
             fig.add_trace(go.Scatter(x=market_df.index, y=market_df['sma_fast'], mode='lines', name='SMA Fast', line=dict(width=1)), row=1, col=1)
        if 'sma_slow' in market_df.columns:
             fig.add_trace(go.Scatter(x=market_df.index, y=market_df['sma_slow'], mode='lines', name='SMA Slow', line=dict(width=1)), row=1, col=1)
        # Add other indicators (EMA, RSI, MACD) similarly, potentially on separate subplots if needed

        fig.update_layout(
            title=f'{selected_symbol} Price Chart ({selected_timeframe})',
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False, # Hide range slider
            legend_title="Indicators"
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    # --- Sentiment Tab Callbacks ---
    @app.callback(
        Output('sentiment-agg-chart', 'figure'),
        [Input('sentiment-timeframe-dropdown', 'value')]
    )
    def update_sentiment_chart(selected_timeframe):
        """Updates the aggregated sentiment chart."""
        log.debug(f"Updating sentiment chart for timeframe {selected_timeframe}")
        sentiment_df = data_provider.get_sentiment_data(selected_timeframe)

        if sentiment_df.empty:
            return create_empty_figure("No sentiment data found")

        # Example: Resample to hourly average sentiment
        # Ensure index is datetime before resampling
        if not pd.api.types.is_datetime64_any_dtype(sentiment_df.index):
             sentiment_df.index = pd.to_datetime(sentiment_df.index, utc=True)

        # Updated to use '1h' instead of '1H' to avoid deprecation warning
        sentiment_resampled = sentiment_df['sentiment_score'].resample('1h').mean() # Hourly average

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sentiment_resampled.index, y=sentiment_resampled, mode='lines', name='Avg Sentiment Score (Hourly)'))
        fig.update_layout(title=f"Average Sentiment Score ({selected_timeframe})", xaxis_title="Time", yaxis_title="Avg Score")

        return fig


    # --- Settings Tab Callbacks ---
    # WARNING: Modifying config module directly at runtime from a web app is generally
    # not recommended, especially in multi-process deployments (like gunicorn).
    # Changes might not persist or affect the running trader process reliably.
    # A better approach involves a dedicated config management system, database flags,
    # or inter-process communication (e.g., Redis, message queue).
    # These callbacks are provided as a basic example but use with extreme caution.

    @app.callback(
        Output('settings-update-capital-pct-status', 'children'),
        [Input('settings-update-capital-pct-button', 'n_clicks')],
        [State('settings-capital-pct-input', 'value')],
        prevent_initial_call=True
    )
    def update_capital_pct(n_clicks, value):
        if n_clicks is None or value is None: raise PreventUpdate
        try:
            new_value = float(value)
            if 0.001 <= new_value <= 0.1: # Basic validation
                log.warning(f"Attempting to update TRADE_CAPITAL_PERCENTAGE to {new_value} (Use with caution!)")
                config.TRADE_CAPITAL_PERCENTAGE = new_value
                return dbc.Alert(f"Capital % updated to {new_value:.3f}", color="success", duration=3000)
            else:
                return dbc.Alert("Invalid value (must be 0.001-0.1)", color="danger", duration=3000)
        except Exception as e:
            log.error(f"Error updating capital %: {e}")
            return dbc.Alert("Update failed", color="danger", duration=3000)

    # Similar callbacks for stop-loss, take-profit, drawdown...
    @app.callback(
        Output('settings-update-stop-loss-status', 'children'),
        [Input('settings-update-stop-loss-button', 'n_clicks')],
        [State('settings-stop-loss-input', 'value')],
        prevent_initial_call=True
    )
    def update_stop_loss(n_clicks, value):
        # Add validation and update config.STOP_LOSS_PCT
        if n_clicks is None or value is None: raise PreventUpdate
        try:
            new_value = float(value)
            if 0.001 <= new_value <= 0.9: # Basic validation
                log.warning(f"Attempting to update STOP_LOSS_PCT to {new_value} (Use with caution!)")
                config.STOP_LOSS_PCT = new_value
                return dbc.Alert(f"Stop Loss % updated to {new_value:.3f}", color="success", duration=3000)
            else:
                return dbc.Alert("Invalid value", color="danger", duration=3000)
        except Exception as e:
            log.error(f"Error updating stop loss %: {e}")
            return dbc.Alert("Update failed", color="danger", duration=3000)


    @app.callback(
        Output('settings-update-take-profit-status', 'children'),
        [Input('settings-update-take-profit-button', 'n_clicks')],
        [State('settings-take-profit-input', 'value')],
        prevent_initial_call=True
    )
    def update_take_profit(n_clicks, value):
        # Add validation and update config.TAKE_PROFIT_PCT
        if n_clicks is None or value is None: raise PreventUpdate
        try:
            new_value = float(value)
            if 0.001 <= new_value <= 1.0: # Basic validation
                log.warning(f"Attempting to update TAKE_PROFIT_PCT to {new_value} (Use with caution!)")
                config.TAKE_PROFIT_PCT = new_value
                return dbc.Alert(f"Take Profit % updated to {new_value:.3f}", color="success", duration=3000)
            else:
                return dbc.Alert("Invalid value", color="danger", duration=3000)
        except Exception as e:
            log.error(f"Error updating take profit %: {e}")
            return dbc.Alert("Update failed", color="danger", duration=3000)

    @app.callback(
        Output('settings-update-drawdown-status', 'children'),
        [Input('settings-update-drawdown-button', 'n_clicks')],
        [State('settings-drawdown-input', 'value')],
        prevent_initial_call=True
    )
    def update_drawdown(n_clicks, value):
         # Add validation and update config.PORTFOLIO_DRAWDOWN_PCT
        if n_clicks is None or value is None: raise PreventUpdate
        try:
            new_value = float(value)
            if 0.01 <= new_value <= 0.99: # Basic validation
                log.warning(f"Attempting to update PORTFOLIO_DRAWDOWN_PCT to {new_value} (Use with caution!)")
                config.PORTFOLIO_DRAWDOWN_PCT = new_value
                return dbc.Alert(f"Max Drawdown % updated to {new_value:.2f}", color="success", duration=3000)
            else:
                return dbc.Alert("Invalid value", color="danger", duration=3000)
        except Exception as e:
            log.error(f"Error updating drawdown %: {e}")
            return dbc.Alert("Update failed", color="danger", duration=3000)


    @app.callback(
        Output('settings-trading-mode-status', 'children'),
        [Input('settings-trading-mode-switch', 'value')],
        prevent_initial_call=True
    )
    def update_trading_mode(live_mode_enabled):
        # WARNING: Changing trading mode requires robust state management.
        # This callback only updates the display and logs a warning.
        # The actual trading_mode needs to be read by the trader logic from a reliable source (state file, DB flag).
        mode = "LIVE" if live_mode_enabled else "PAPER"
        log.warning(f"Dashboard switch toggled to {mode}. Actual trading mode depends on trader process reading shared state mechanism - NOT YET IMPLEMENTED)")
        # TODO: Implement mechanism to signal trading mode change to the scheduler/trader process
        return f"Current Mode Display: {mode} (Control requires shared state mechanism - NOT YET IMPLEMENTED)"


    @app.callback(
        Output('settings-reenable-symbol-status', 'children'),
        [Input('settings-reenable-symbol-button', 'n_clicks')],
        [State('settings-disabled-symbol-dropdown', 'value')],
        prevent_initial_call=True
    )
    def reenable_symbol(n_clicks, symbol_to_enable):
        if n_clicks is None or not symbol_to_enable:
            raise PreventUpdate
        try:
            log.info(f"Dashboard request to re-enable symbol: {symbol_to_enable}")
            trader.enable_symbol(symbol_to_enable) # Call function in trader module
            return dbc.Alert(f"Symbol {symbol_to_enable} re-enabled.", color="success", duration=3000)
        except Exception as e:
            log.error(f"Error re-enabling symbol {symbol_to_enable}: {e}", exc_info=True)
            return dbc.Alert(f"Failed to re-enable {symbol_to_enable}.", color="danger", duration=3000)


    log.info("Dashboard callbacks registered.")


# --- Helper function used in callbacks ---
def create_empty_figure(message="No data available") -> go.Figure:
    """Creates an empty Plotly figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': message, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
    )
    return fig
