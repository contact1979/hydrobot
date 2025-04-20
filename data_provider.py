# /dashboard/data_provider.py

import logging
import pandas as pd
import datetime
import pytz
from sqlalchemy.sql import select, desc, text
from typing import Dict, List, Optional, Any, Tuple

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils # Import the db_utils module to access engine, table objects and session
from trading import trader # To get disabled symbols list
# from .. import config
# from ..database import db_utils
# from ..trading import trader # Careful with circular imports if trader imports portfolio which might use DB

log = logging.getLogger(__name__)

# --- Data Fetching Functions for Dashboard ---

def get_overview_data() -> Dict[str, Any]:
    """Fetches key metrics for the overview section."""
    data = {
        "pnl_total": 0.0, # TODO: Calculate overall PnL (needs careful tracking)
        "open_positions_count": 0, # TODO: Get from portfolio manager state or DB
        "cash": config.INITIAL_CAPITAL_USD, # TODO: Get current cash from portfolio manager state
        "total_value": config.INITIAL_CAPITAL_USD, # TODO: Get from portfolio manager state
        "peak_value": config.INITIAL_CAPITAL_USD, # TODO: Get from portfolio manager state
        "halt_status": False, # TODO: Get from portfolio manager state
        "latest_trades": pd.DataFrame(),
        "error_alerts": [] # TODO: Implement system for tracking/displaying errors
    }
    if not db_utils.engine:
        log.error("Database engine not available for get_overview_data.")
        return data # Return defaults if DB not configured

    try:
        # Use the engine directly with pd.read_sql when passing SQLAlchemy selectables
        engine = db_utils.engine
        trade_query = select(db_utils.trade_log).order_by(desc(db_utils.trade_log.c.timestamp)).limit(10)
        data['latest_trades'] = pd.read_sql(trade_query, engine, parse_dates=['timestamp']) # Use engine

        # Convert timestamp to user's local timezone for display? Or keep UTC? Let's keep UTC for now.
        if not data['latest_trades'].empty and data['latest_trades']['timestamp'].dt.tz is None:
             # Ensure timezone aware if read directly without timezone info
             data['latest_trades']['timestamp'] = data['latest_trades']['timestamp'].dt.tz_localize(pytz.utc)
        elif not data['latest_trades'].empty:
             data['latest_trades']['timestamp'] = data['latest_trades']['timestamp'].dt.tz_convert(pytz.utc) # Ensure UTC
        # data['latest_trades']['timestamp'] = data['latest_trades']['timestamp'].dt.tz_convert('America/Los_Angeles') # Example conversion

        # TODO: Query portfolio state if persisted in DB, otherwise needs access to live PortfolioManager instance
        # This is a challenge for a separate dashboard process. Might need IPC or a shared state DB.
        # For now, returning defaults or placeholders.

    except Exception as e:
        log.error(f"Error fetching overview data from DB: {e}", exc_info=True)

    # Get disabled symbols from trader module (in-memory state)
    # Ensure trader module is imported and state is accessible
    try:
        data["disabled_symbols"] = list(trader.DISABLED_SYMBOLS)
    except AttributeError: # Handle case where trader or DISABLED_SYMBOLS might not be initialized yet
        log.warning("Could not access trader.DISABLED_SYMBOLS state.")
        data["disabled_symbols"] = []

    # Get halt status from portfolio manager (needs access to the instance)
    # data["halt_status"] = portfolio_manager_instance.halt_trading_flag # Requires instance access

    return data


def get_market_data(symbol: str, time_window: str = '24h') -> pd.DataFrame:
    """Fetches price/TA data for a specific symbol and time window."""
    if not symbol:
        return pd.DataFrame()
    if not db_utils.engine:
        log.error(f"Database engine not available for get_market_data({symbol}).")
        return pd.DataFrame()

    end_time = datetime.datetime.now(pytz.utc)
    try:
        # Convert time_window string (e.g., '1h', '4h', '24h', '7d') to Timedelta
        delta = pd.Timedelta(time_window)
        start_time = end_time - delta
    except Exception:
        log.warning(f"Invalid time window format: {time_window}. Defaulting to 24h.")
        start_time = end_time - pd.Timedelta(hours=24)

    market_df = pd.DataFrame()
    try:
        engine = db_utils.engine
        price_query = select(db_utils.price_data).where(
            db_utils.price_data.c.symbol == symbol,
            db_utils.price_data.c.interval == '5m', # Assuming we display 5m candles
            db_utils.price_data.c.open_time >= start_time,
            db_utils.price_data.c.open_time <= end_time
        ).order_by(db_utils.price_data.c.open_time)
        market_df = pd.read_sql(price_query, engine, index_col='open_time', parse_dates=['open_time']) # Use engine

        if not market_df.empty and market_df.index.tz is None:
             # Ensure index is timezone-aware UTC if read without timezone info
             market_df.index = market_df.index.tz_localize(pytz.utc)
        elif not market_df.empty:
             market_df.index = market_df.index.tz_convert(pytz.utc) # Ensure UTC

        log.debug(f"Fetched {len(market_df)} market data points for {symbol} ({time_window}).")
    except Exception as e:
        log.error(f"Error fetching market data for {symbol} from DB: {e}", exc_info=True)

    return market_df


def get_sentiment_data(time_window: str = '24h') -> pd.DataFrame:
    """Fetches sentiment analysis results over a time window."""
    if not db_utils.engine:
        log.error("Database engine not available for get_sentiment_data.")
        return pd.DataFrame()

    end_time = datetime.datetime.now(pytz.utc)
    try:
        delta = pd.Timedelta(time_window)
        start_time = end_time - delta
    except Exception:
        log.warning(f"Invalid time window format: {time_window}. Defaulting to 24h.")
        start_time = end_time - pd.Timedelta(hours=24)

    sentiment_df = pd.DataFrame()
    try:
        engine = db_utils.engine
        # Fetch sentiment scores over time
        # TODO: Add joins to filter by symbol if needed/possible
        sentiment_query = select(
            db_utils.sentiment_analysis_results.c.analyzed_at,
            db_utils.sentiment_analysis_results.c.sentiment_score,
            # Include source if needed for breakdown (requires join)
            # db_utils.news_data.c.source_publisher.label('source') # Example join
        ).where(
            db_utils.sentiment_analysis_results.c.analyzed_at >= start_time,
            db_utils.sentiment_analysis_results.c.analyzed_at <= end_time
        ).order_by(db_utils.sentiment_analysis_results.c.analyzed_at)
        # .select_from(db_utils.sentiment_analysis_results.join(db_utils.news_data, db_utils.sentiment_analysis_results.c.news_id == db_utils.news_data.c.id, isouter=True)) # Example join structure

        sentiment_df = pd.read_sql(sentiment_query, engine, index_col='analyzed_at', parse_dates=['analyzed_at']) # Use engine

        if not sentiment_df.empty and sentiment_df.index.tz is None:
             # Ensure index is timezone-aware UTC if read without timezone info
             sentiment_df.index = sentiment_df.index.tz_localize(pytz.utc)
        elif not sentiment_df.empty:
             sentiment_df.index = sentiment_df.index.tz_convert(pytz.utc) # Ensure UTC

        log.debug(f"Fetched {len(sentiment_df)} sentiment data points ({time_window}).")

        # TODO: Fetch recent raw posts/tweets/news (e.g., last 10-20) for display

    except Exception as e:
        log.error(f"Error fetching sentiment data from DB: {e}", exc_info=True)

    return sentiment_df

def get_trading_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches open positions and trade history."""
    open_positions_df = pd.DataFrame(columns=['symbol', 'quantity', 'entry_price', 'entry_time']) # Placeholder
    trade_history_df = pd.DataFrame()

    # TODO: Get open positions from live PortfolioManager instance or persisted state
    # This is a placeholder implementation fetching from trade log (not accurate for open positions)
    # open_positions = portfolio_manager_instance.get_all_positions()
    # open_positions_df = pd.DataFrame.from_dict(open_positions, orient='index')

    if not db_utils.engine:
        log.error("Database engine not available for get_trading_data.")
        return open_positions_df, trade_history_df

    try:
        engine = db_utils.engine
        # Fetch recent trade history
        history_query = select(db_utils.trade_log).order_by(desc(db_utils.trade_log.c.timestamp)).limit(100) # Fetch last 100 trades
        trade_history_df = pd.read_sql(history_query, engine, parse_dates=['timestamp']) # Use engine

        if not trade_history_df.empty and trade_history_df['timestamp'].dt.tz is None:
             # Ensure timezone aware if read directly without timezone info
             trade_history_df['timestamp'] = trade_history_df['timestamp'].dt.tz_localize(pytz.utc)
        elif not trade_history_df.empty:
             trade_history_df['timestamp'] = trade_history_df['timestamp'].dt.tz_convert(pytz.utc) # Ensure UTC

        log.debug(f"Fetched {len(trade_history_df)} trade history records.")

    except Exception as e:
        log.error(f"Error fetching trading data from DB: {e}", exc_info=True)

    return open_positions_df, trade_history_df


def get_performance_data() -> pd.DataFrame:
    """Fetches data needed for performance analysis (e.g., PnL over time)."""
    # This could involve fetching all 'SELL' trades and calculating cumulative PnL
    pnl_df = pd.DataFrame()
    if not db_utils.engine:
        log.error("Database engine not available for get_performance_data.")
        return pnl_df

    try:
        engine = db_utils.engine
        pnl_query = select(
                db_utils.trade_log.c.timestamp,
                db_utils.trade_log.c.pnl
            ).where(
                db_utils.trade_log.c.trade_type == 'SELL',
                db_utils.trade_log.c.pnl.isnot(None) # Ensure PnL is calculated
            ).order_by(db_utils.trade_log.c.timestamp)
        pnl_df = pd.read_sql(pnl_query, engine, index_col='timestamp', parse_dates=['timestamp']) # Use engine

        if not pnl_df.empty and pnl_df.index.tz is None:
             # Ensure index is timezone-aware UTC if read without timezone info
             pnl_df.index = pnl_df.index.tz_localize(pytz.utc)
        elif not pnl_df.empty:
             pnl_df.index = pnl_df.index.tz_convert(pytz.utc) # Ensure UTC
             pnl_df['cumulative_pnl'] = pnl_df['pnl'].cumsum()
        log.debug(f"Fetched {len(pnl_df)} PnL records for performance analysis.")
    except Exception as e:
        log.error(f"Error fetching performance data from DB: {e}", exc_info=True)
    return pnl_df


def get_config_settings() -> Dict[str, Any]:
    """Returns current trading configuration settings."""
    # Reads directly from the (potentially updated at runtime) config module
    # NOTE: This assumes the dashboard process shares the same memory space
    # or can somehow access the potentially modified config values.
    # This might not work reliably if trader/dashboard are separate processes.
    try:
        return {
            "trade_capital_percentage": config.TRADE_CAPITAL_PERCENTAGE,
            "stop_loss_pct": config.STOP_LOSS_PCT,
            "take_profit_pct": config.TAKE_PROFIT_PCT,
            "portfolio_drawdown_pct": config.PORTFOLIO_DRAWDOWN_PCT,
            # Add other relevant config values
        }
    except Exception as e:
        log.error(f"Error reading config settings: {e}")
        return {} # Return empty dict on error

def get_target_symbols_list() -> List[str]:
     """Gets the list of symbols the bot might trade."""
     # This could be dynamic based on get_target_symbols or a fixed list
     # For dropdowns, maybe fetch all symbols with recent price data?
     symbols = []
     if not db_utils.engine:
         log.error("Database engine not available for get_target_symbols_list.")
         return ['BTCUSDT', 'ETHUSDT'] # Fallback

     try:
         engine = db_utils.engine
         # Query distinct symbols from recent price data
         query = select(db_utils.price_data.c.symbol.distinct()).where(
             db_utils.price_data.c.open_time >= datetime.datetime.now(pytz.utc) - pd.Timedelta(days=1)
         ).order_by(db_utils.price_data.c.symbol)
         # Use engine for connection in read_sql
         with engine.connect() as connection:
             symbols_df = pd.read_sql(query, connection) # Use connection from engine
             symbols = symbols_df['symbol'].tolist()
     except Exception as e:
         log.error(f"Error fetching distinct symbols from DB: {e}", exc_info=True)
     return symbols if symbols else ['BTCUSDT', 'ETHUSDT'] # Fallback


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')
    print("--- Testing Dashboard Data Provider ---")

    # Ensure DB is available for testing
    if not db_utils.engine:
         print("DB Engine not configured. Cannot run tests.")
    else:
        print("\nFetching overview data...")
        overview = get_overview_data()
        print(f"Overview Keys: {overview.keys()}")
        print(f"Latest Trades DF Shape: {overview.get('latest_trades', pd.DataFrame()).shape}")
        print(f"Disabled Symbols: {overview.get('disabled_symbols')}")

        print("\nFetching market data for BTCUSDT (1h)...")
        market = get_market_data('BTCUSDT', '1h')
        print(f"Market DF Shape: {market.shape}")
        if not market.empty: print(market.head())

        print("\nFetching sentiment data (4h)...")
        sentiment = get_sentiment_data('4h')
        print(f"Sentiment DF Shape: {sentiment.shape}")
        if not sentiment.empty: print(sentiment.head())

        print("\nFetching trading data...")
        open_pos, history = get_trading_data()
        print(f"Open Positions DF Shape: {open_pos.shape}")
        print(f"Trade History DF Shape: {history.shape}")
        if not history.empty: print(history.head())

        print("\nFetching performance data...")
        perf = get_performance_data()
        print(f"Performance DF Shape: {perf.shape}")
        if not perf.empty: print(perf.head())

        print("\nFetching config settings...")
        settings = get_config_settings()
        print(settings)

        print("\nFetching target symbols list...")
        symbols_list = get_target_symbols_list()
        print(symbols_list)


    print("\n--- Test Complete ---")

