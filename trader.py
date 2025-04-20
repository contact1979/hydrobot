# /trading/trader.py

import logging
import time
import pandas as pd
import pytz
from typing import Dict, Optional, Tuple, List
from binance.client import Client # Import SIDE_BUY, SIDE_SELL etc.
from binance.exceptions import BinanceAPIException, BinanceOrderException, BinanceOrderMinAmountException, BinanceOrderMinPriceException, BinanceOrderMinTotalException

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils
from data_collection import binance_client # Needs function to get current prices
from modeling import predictor # Needs make_prediction function
from trading.portfolio import PortfolioManager # Import the manager class
# from .. import config
# from ..database import db_utils
# from ..data_collection import binance_client
# from ..modeling import predictor
# from .portfolio import PortfolioManager

log = logging.getLogger(__name__)

# --- Global State (Potentially move to a dedicated state manager) ---
# Track symbols temporarily disabled due to errors
DISABLED_SYMBOLS = set()
# Track portfolio state (in-memory for now, could be loaded/persisted)
# Initialize portfolio manager globally or pass it around
# For simplicity, assume it's passed to execute_trade_logic
# portfolio_manager = PortfolioManager() # Or load from saved state

# --- Helper Functions ---

def _get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetches current market prices for a list of symbols."""
    client = binance_client.get_binance_client()
    if not client or not symbols:
        return {}

    prices = {}
    log.debug(f"Fetching current prices for {len(symbols)} symbols...")
    try:
        tickers = client.get_symbol_ticker() # Get all tickers
        ticker_map = {t['symbol']: float(t['price']) for t in tickers if t.get('price')}
        for symbol in symbols:
            if symbol in ticker_map:
                prices[symbol] = ticker_map[symbol]
            else:
                log.warning(f"Could not find current price for {symbol} in ticker list.")
    except Exception as e:
        log.error(f"Error fetching current prices from Binance: {e}", exc_info=True)
    return prices


def _place_order(symbol: str, side: str, quantity: float, trading_mode: str, retry_count: int = 3) -> Tuple[bool, Dict]:
    """
    Places a MARKET order (simulated or live) and handles retries.

    Args:
        symbol (str): Trading symbol.
        side (str): 'BUY' or 'SELL'.
        quantity (float): Quantity to trade.
        trading_mode (str): 'PAPER' or 'LIVE'.
        retry_count (int): Number of times to retry on failure.

    Returns:
        Tuple[bool, Dict]: (success_status, order_details)
            order_details contains keys like 'orderId', 'status', 'executedQty', 'avgPrice', 'fee', etc.
    """
    log.info(f"Attempting to place {trading_mode} {side} order: {quantity:.8f} {symbol}")
    order_details = {
        'symbol': symbol,
        'side': side,
        'type': 'MARKET',
        'quantity': quantity,
        'status': 'FAILED', # Default status
        'orderId': None,
        'executedQty': 0.0,
        'avgPrice': 0.0,
        'fee': 0.0,
        'timestamp': pd.Timestamp.now(tz=pytz.utc)
    }

    if trading_mode == 'LIVE':
        client = binance_client.get_binance_client()
        if not client:
            log.error("LIVE Trading Error: Binance client not available.")
            return False, order_details

        for attempt in range(retry_count):
            try:
                # --- Execute Live Order ---
                order_func = client.create_order if side == Client.SIDE_BUY else client.create_order # Use same func for BUY/SELL market
                # Note: create_order might need different params based on order type
                # For MARKET orders, quantity is usually required for BUY/SELL on SPOT
                log.info(f"Executing LIVE {side} order for {quantity:.8f} {symbol} (Attempt {attempt + 1}/{retry_count})")
                live_order = order_func(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                    # For MARKET SELL using quote qty: quoteOrderQty=usd_value
                )
                log.info(f"LIVE Order Response: {live_order}")

                # --- Parse Response ---
                order_details['status'] = live_order.get('status', 'UNKNOWN')
                order_details['orderId'] = live_order.get('orderId')
                order_details['timestamp'] = pd.to_datetime(live_order.get('transactTime'), unit='ms', utc=True) if live_order.get('transactTime') else order_details['timestamp']

                if order_details['status'] == Client.ORDER_STATUS_FILLED:
                    order_details['executedQty'] = float(live_order.get('executedQty', 0.0))
                    cummulative_quote_qty = float(live_order.get('cummulativeQuoteQty', 0.0))
                    if order_details['executedQty'] > 1e-9:
                        order_details['avgPrice'] = cummulative_quote_qty / order_details['executedQty']
                    else:
                         order_details['avgPrice'] = 0.0 # Avoid division by zero

                    # TODO: Calculate fees correctly based on 'fills' part of response if available
                    order_details['fee'] = 0.0 # Placeholder

                    log.info(f"LIVE {side} order FILLED: {order_details['executedQty']:.8f} {symbol} @ avg price {order_details['avgPrice']:.4f}")
                    return True, order_details
                else:
                    # Handle partially filled or other non-failed statuses if necessary
                    log.warning(f"LIVE {side} order status is '{order_details['status']}'. Order: {live_order}")
                    # Treat non-FILLED as failure for now unless specific handling is added
                    # return False, order_details # Or maybe True depending on status? Needs careful thought.
                    # For simplicity, assume only FILLED is full success for market orders here.
                    order_details['status'] = 'FAILED' # Mark as failed if not fully filled immediately
                    return False, order_details

            except (BinanceOrderMinAmountException, BinanceOrderMinPriceException, BinanceOrderMinTotalException) as e:
                 log.error(f"LIVE Order Failed (Min Amount/Price/Total): {e}. Symbol: {symbol}, Qty: {quantity}. No retry.")
                 order_details['status'] = 'REJECTED_MIN_REQ'
                 return False, order_details # Don't retry if order size is invalid
            except BinanceOrderException as e:
                 log.error(f"LIVE Order Failed (BinanceOrderException): {e}. Symbol: {symbol}, Qty: {quantity}. No retry.")
                 order_details['status'] = 'REJECTED_BINANCE'
                 return False, order_details # Don't retry known order rejections
            except (BinanceAPIException, BinanceRequestException) as e:
                log.warning(f"LIVE Order Attempt {attempt + 1}/{retry_count} failed (API/Request Error): {e}. Retrying...")
                time.sleep(2 ** attempt) # Exponential backoff
            except Exception as e:
                log.error(f"LIVE Order Attempt {attempt + 1}/{retry_count} failed (Unexpected Error): {e}", exc_info=True)
                time.sleep(2 ** attempt) # Exponential backoff

        # If loop finishes without success
        log.error(f"LIVE {side} order failed after {retry_count} attempts for {symbol}.")
        return False, order_details

    elif trading_mode == 'PAPER':
        # --- Simulate Paper Order ---
        # Assume immediate fill at current price (or add simulated slippage)
        current_price = _get_current_prices([symbol]).get(symbol)
        if current_price is None:
             log.error(f"PAPER Trading Error: Cannot get current price for {symbol} to simulate fill.")
             return False, order_details

        # Simulate slippage (optional)
        # slippage_factor = config.SLIPPAGE_RATE # e.g., 0.001 for 0.1%
        # if side == Client.SIDE_BUY:
        #     fill_price = current_price * (1 + slippage_factor)
        # else: # SELL
        #     fill_price = current_price * (1 - slippage_factor)
        fill_price = current_price # Simple simulation

        order_details['status'] = 'SIMULATED_FILLED'
        order_details['orderId'] = f"PAPER_{int(time.time()*1000)}" # Generate dummy ID
        order_details['executedQty'] = quantity
        order_details['avgPrice'] = fill_price
        # Simulate fees (optional)
        # order_details['fee'] = quantity * fill_price * config.FEE_RATE
        order_details['fee'] = 0.0

        log.info(f"PAPER {side} order FILLED: {order_details['executedQty']:.8f} {symbol} @ avg price {order_details['avgPrice']:.4f}")
        return True, order_details
    else:
        log.error(f"Invalid trading_mode: {trading_mode}")
        return False, order_details


def check_exit_conditions(symbol: str, current_price: float, position: Dict, stop_loss_pct: float, take_profit_pct: float) -> Optional[str]:
    """
    Checks if stop-loss or take-profit conditions are met for a position.

    Args:
        symbol (str): Symbol being checked.
        current_price (float): Current market price.
        position (Dict): Dictionary containing 'entry_price' and 'quantity'.
        stop_loss_pct (float): Stop loss percentage (e.g., 0.05).
        take_profit_pct (float): Take profit percentage (e.g., 0.10).

    Returns:
        Optional[str]: 'STOP_LOSS', 'TAKE_PROFIT', or None.
    """
    if not position or current_price <= 0:
        return None

    entry_price = position['entry_price']
    if entry_price <= 0: # Should not happen with valid positions
        return None

    # Stop Loss Check
    stop_loss_price = entry_price * (1 - stop_loss_pct)
    if current_price <= stop_loss_price:
        log.info(f"EXIT condition met for {symbol}: STOP LOSS triggered (Current: {current_price:.4f} <= SL: {stop_loss_price:.4f})")
        return 'STOP_LOSS'

    # Take Profit Check
    take_profit_price = entry_price * (1 + take_profit_pct)
    if current_price >= take_profit_price:
        log.info(f"EXIT condition met for {symbol}: TAKE PROFIT triggered (Current: {current_price:.4f} >= TP: {take_profit_price:.4f})")
        return 'TAKE_PROFIT'

    return None


def liquidate_position(portfolio: PortfolioManager, symbol: str, current_price: float, trading_mode: str, reason: str) -> bool:
    """Attempts to liquidate (sell) the entire position for a given symbol."""
    position = portfolio.get_position(symbol)
    if not position or position['quantity'] <= 0:
        log.info(f"No position to liquidate for {symbol}.")
        return True # Nothing to do

    sell_qty = position['quantity']
    log.warning(f"Attempting to liquidate position for {symbol} ({sell_qty:.8f}) due to: {reason}")

    success, order_details = _place_order(symbol, Client.SIDE_SELL, sell_qty, trading_mode)

    if success:
        # Calculate realized PnL
        entry_price = position['entry_price']
        pnl = (order_details['avgPrice'] - entry_price) * order_details['executedQty']

        # Update portfolio state
        portfolio.update_position(symbol, -order_details['executedQty'], order_details['avgPrice'], order_details['timestamp'])

        # Log the liquidation trade to DB
        log_trade_to_db(
            symbol=symbol,
            trade_type=Client.SIDE_SELL,
            order_type='MARKET',
            status=order_details['status'],
            binance_order_id=order_details.get('orderId'),
            price=order_details['avgPrice'],
            quantity=order_details['executedQty'],
            fee=order_details['fee'],
            pnl=pnl,
            signal_confidence=None, # No model confidence for liquidation
            trigger_reason=reason,
            trading_mode=trading_mode
        )
        return True
    else:
        log.error(f"Failed to liquidate position for {symbol}. Reason: {reason}. Order Status: {order_details.get('status')}")
        # Potentially retry or raise alert
        return False


def log_trade_to_db(**kwargs):
    """Logs trade details to the trade_log table in the database."""
    trade_record = {
        'timestamp': pd.Timestamp.now(tz=pytz.utc), # Log time
        'symbol': kwargs.get('symbol'),
        'trade_type': kwargs.get('trade_type'),
        'order_type': kwargs.get('order_type', 'MARKET'),
        'status': kwargs.get('status'),
        'binance_order_id': kwargs.get('binance_order_id'),
        'price': kwargs.get('price'),
        'quantity': kwargs.get('quantity'),
        'usd_value': kwargs.get('price', 0) * kwargs.get('quantity', 0),
        'fee': kwargs.get('fee'),
        'pnl': kwargs.get('pnl'),
        'signal_confidence': kwargs.get('signal_confidence'),
        'trigger_reason': kwargs.get('trigger_reason'),
        'trading_mode': kwargs.get('trading_mode')
    }
    # Filter out None values before insertion if DB schema requires it
    trade_record_clean = {k: v for k, v in trade_record.items() if v is not None}

    try:
        db_utils.bulk_insert_data([trade_record_clean], db_utils.trade_log) # Use bulk insert for single record
        log.info(f"Trade logged to DB: {trade_record_clean.get('trade_type')} {trade_record_clean.get('quantity')} {trade_record_clean.get('symbol')}")
    except Exception as e:
        log.error(f"Failed to log trade to database: {e}", exc_info=True)
        # Consider fallback logging (e.g., to file) if DB logging fails


def execute_trade_cycle(portfolio: PortfolioManager, trading_mode: str):
    """
    Executes one cycle of the trading logic: fetches predictions, checks positions,
    places orders based on signals and risk rules.

    Args:
        portfolio (PortfolioManager): The portfolio manager instance.
        trading_mode (str): 'PAPER' or 'LIVE'.
    """
    log.info(f"--- Starting Trade Cycle (Mode: {trading_mode}) ---")

    # --- 0. Check Global Halt Conditions ---
    if portfolio.halt_trading_flag:
        log.warning("Trading HALTED due to maximum drawdown breach. No new trades will be placed.")
        # We might still need to manage existing positions or allow manual intervention
        return # Skip the rest of the cycle

    # --- 1. Get Symbols and Latest Data ---
    # Define symbols to check (e.g., all currently monitored low-price symbols)
    # For now, let's assume we get target symbols dynamically
    try:
        symbols_to_monitor = binance_client.get_target_symbols()
        if not symbols_to_monitor:
             log.warning("No target symbols found to monitor in this cycle.")
             return
    except Exception as e:
        log.error(f"Failed to get target symbols: {e}", exc_info=True)
        return

    # Add symbols currently in portfolio to ensure they are checked for exits
    symbols_in_portfolio = list(portfolio.get_all_positions().keys())
    all_symbols_to_check = list(set(symbols_to_monitor + symbols_in_portfolio))

    # Filter out disabled symbols
    active_symbols = [s for s in all_symbols_to_check if s not in DISABLED_SYMBOLS]
    if not active_symbols:
         log.info("No active symbols to process in this cycle.")
         return

    log.debug(f"Active symbols for this cycle: {active_symbols}")

    # Get current prices
    current_prices = _get_current_prices(active_symbols)
    if not current_prices:
        log.error("Failed to get current prices. Skipping trade cycle.")
        return

    # --- 2. Update Portfolio Value & Check Drawdown ---
    portfolio.calculate_total_value(current_prices)
    if portfolio.check_drawdown_and_halt():
        # Drawdown breached, initiate liquidation
        log.critical("Drawdown breached! Initiating liquidation of all positions.")
        positions_to_liquidate = list(portfolio.get_all_positions().keys())
        for symbol in positions_to_liquidate:
             price = current_prices.get(symbol)
             if price:
                 liquidate_position(portfolio, symbol, price, trading_mode, reason="MAX_DRAWDOWN")
             else:
                 log.error(f"Cannot liquidate {symbol}: Missing current price.")
        return # Stop further processing after initiating liquidation

    # --- 3. Get Predictions ---
    # We need the latest features for prediction. Assume feature_generator runs just before this.
    # This part needs careful orchestration - how do we get the *single* latest feature row?
    # Simplification: Assume predictor can access latest features or we pass them.
    # Let's assume predictor handles fetching/generating its required input for now.
    predictions: Dict[str, Tuple[int, float]] = {} # {symbol: (prediction, confidence)}
    for symbol in active_symbols:
         try:
             # TODO: Need mechanism to get latest features for prediction input
             # latest_feature_row_df = get_latest_features(symbol) # Placeholder function
             # if latest_feature_row_df is not None:
             #     pred_result = predictor.make_prediction(latest_feature_row_df)
             #     if pred_result:
             #         predictions[symbol] = pred_result
             pass # Placeholder - Prediction logic needs integration with feature generation output
         except Exception as e:
             log.error(f"Failed to get prediction for {symbol}: {e}", exc_info=True)

    # --- 4. Iterate Through Symbols and Apply Logic ---
    open_positions = portfolio.get_all_positions()
    num_open_positions = len(open_positions)

    # TODO: Define confidence thresholds from config or elsewhere
    BUY_CONFIDENCE_THRESHOLD = 0.65 # Example
    SELL_CONFIDENCE_THRESHOLD = 0.65 # Example (for prob of class 1 < (1-threshold))

    for symbol in active_symbols:
        current_price = current_prices.get(symbol)
        if not current_price:
            log.warning(f"Skipping {symbol}: No current price available.")
            continue

        position = portfolio.get_position(symbol)
        prediction_result = predictions.get(symbol) # (pred_class, prob_class_1)

        # --- Exit Logic ---
        if position:
            exit_reason = check_exit_conditions(
                symbol, current_price, position,
                config.STOP_LOSS_PCT, config.TAKE_PROFIT_PCT
            )
            should_sell = False
            trigger = exit_reason # Default trigger is SL/TP

            if exit_reason:
                should_sell = True
            elif prediction_result:
                pred_class, prob_class_1 = prediction_result
                # Check for strong SELL signal (e.g., pred_class=0 and high confidence)
                prob_class_0 = 1.0 - prob_class_1
                if pred_class == 0 and prob_class_0 >= SELL_CONFIDENCE_THRESHOLD:
                    log.info(f"EXIT condition met for {symbol}: Strong SELL signal (Prob: {prob_class_0:.4f})")
                    should_sell = True
                    trigger = 'MODEL_SIGNAL_SELL'

            if should_sell:
                sell_qty = position['quantity']
                success, order_details = _place_order(symbol, Client.SIDE_SELL, sell_qty, trading_mode)
                if success:
                    entry_price = position['entry_price']
                    pnl = (order_details['avgPrice'] - entry_price) * order_details['executedQty']
                    portfolio.update_position(symbol, -order_details['executedQty'], order_details['avgPrice'], order_details['timestamp'])
                    log_trade_to_db(
                        symbol=symbol, trade_type=Client.SIDE_SELL, status=order_details['status'],
                        binance_order_id=order_details.get('orderId'), price=order_details['avgPrice'],
                        quantity=order_details['executedQty'], fee=order_details['fee'], pnl=pnl,
                        signal_confidence=prob_class_0 if trigger == 'MODEL_SIGNAL_SELL' else None,
                        trigger_reason=trigger, trading_mode=trading_mode
                    )
                else: # Sell order failed
                     log.error(f"Failed to execute {trigger} SELL order for {symbol}. Status: {order_details.get('status')}")
                     # Decide how to handle failed exits - retry later? Manual alert? Disable symbol?
                     if order_details.get('status') in ['REJECTED_MIN_REQ', 'REJECTED_BINANCE']: # Non-retryable
                          log.warning(f"Disabling symbol {symbol} due to non-retryable order rejection.")
                          DISABLED_SYMBOLS.add(symbol)
                     # If retryable error occurred here after retries in _place_order, maybe disable temporarily
                     elif trading_mode == 'LIVE': # Only disable on live failures after retries
                          log.warning(f"Disabling symbol {symbol} temporarily due to repeated API errors during SELL.")
                          DISABLED_SYMBOLS.add(symbol)
                          # Dashboard should show disabled symbols and allow re-enabling

        # --- Entry Logic ---
        elif not position: # Only consider entry if not currently holding
            if num_open_positions >= config.MAX_CONCURRENT_POSITIONS:
                log.debug(f"Skipping potential entry for {symbol}: Max concurrent positions ({config.MAX_CONCURRENT_POSITIONS}) reached.")
                continue # Skip entry if limit reached

            if prediction_result:
                pred_class, prob_class_1 = prediction_result
                # Check for strong BUY signal
                if pred_class == 1 and prob_class_1 >= BUY_CONFIDENCE_THRESHOLD:
                    log.info(f"ENTRY signal detected for {symbol}: Strong BUY signal (Prob: {prob_class_1:.4f})")

                    # Calculate quantity based on % capital
                    buy_qty = portfolio.get_trade_quantity(symbol, current_price, config.TRADE_CAPITAL_PERCENTAGE)

                    if buy_qty <= 0:
                        log.warning(f"Skipping BUY for {symbol}: Calculated quantity is zero (check capital, price, min lot size).")
                        continue

                    # Check if enough cash available (redundant if get_trade_quantity uses available capital, but good safeguard)
                    required_cash = buy_qty * current_price
                    if required_cash > portfolio.get_available_capital():
                         log.warning(f"Skipping BUY for {symbol}: Insufficient cash (${portfolio.get_available_capital():.2f}) for trade value (${required_cash:.2f}).")
                         continue

                    # Place BUY order
                    success, order_details = _place_order(symbol, Client.SIDE_BUY, buy_qty, trading_mode)
                    if success:
                        portfolio.update_position(symbol, order_details['executedQty'], order_details['avgPrice'], order_details['timestamp'])
                        log_trade_to_db(
                            symbol=symbol, trade_type=Client.SIDE_BUY, status=order_details['status'],
                            binance_order_id=order_details.get('orderId'), price=order_details['avgPrice'],
                            quantity=order_details['executedQty'], fee=order_details['fee'], pnl=None,
                            signal_confidence=prob_class_1, trigger_reason='MODEL_SIGNAL_BUY', trading_mode=trading_mode
                        )
                        num_open_positions += 1 # Increment count after successful entry
                    else: # Buy order failed
                        log.error(f"Failed to execute BUY order for {symbol}. Status: {order_details.get('status')}")
                        # Handle failed entry - disable symbol? Log?
                        if order_details.get('status') in ['REJECTED_MIN_REQ', 'REJECTED_BINANCE']:
                            log.warning(f"Disabling symbol {symbol} due to non-retryable order rejection.")
                            DISABLED_SYMBOLS.add(symbol)
                        elif trading_mode == 'LIVE':
                            log.warning(f"Disabling symbol {symbol} temporarily due to repeated API errors during BUY.")
                            DISABLED_SYMBOLS.add(symbol)

    log.info(f"--- Trade Cycle Finished ---")


# --- Function to manually re-enable a symbol ---
def enable_symbol(symbol: str):
    """Removes a symbol from the disabled list."""
    if symbol in DISABLED_SYMBOLS:
        log.warning(f"Manually re-enabling trading for symbol: {symbol}")
        DISABLED_SYMBOLS.remove(symbol)
    else:
        log.info(f"Symbol {symbol} is not currently disabled.")


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Trader Logic ---")
    # This requires other modules (predictor, binance_client, feature_generator) to be working
    # and potentially a trained model and database data.
    # Running this standalone is complex. Best tested as part of the main application loop.

    # Example: Initialize portfolio and run one cycle in PAPER mode
    test_portfolio = PortfolioManager(initial_capital=10000.0)
    print("Running one PAPER trade cycle...")
    try:
        execute_trade_cycle(test_portfolio, trading_mode='PAPER')
        print("\nPaper trade cycle finished. Check logs for details.")
        print(f"Final Paper Portfolio Cash: ${test_portfolio.get_available_capital():.2f}")
        print(f"Final Paper Portfolio Positions: {test_portfolio.get_all_positions()}")
    except Exception as e:
        print(f"Error during paper trade cycle test: {e}")
        log.error("Error in trader test run", exc_info=True)

    print("\n--- Test Complete ---")

