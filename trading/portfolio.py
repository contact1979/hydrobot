# /trading/portfolio.py

import logging
import pandas as pd
from typing import Dict, Optional, Tuple

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
# from .. import config

log = logging.getLogger(__name__)

class PortfolioManager:
    """
    Manages the state of the trading portfolio (cash, positions, value).
    Designed primarily for paper trading simulation but can track live state too.
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL_USD):
        """
        Initializes the PortfolioManager.

        Args:
            initial_capital (float): Starting cash balance.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        # Positions format: { 'SYMBOL': {'quantity': float, 'entry_price': float, 'entry_time': datetime} }
        self.positions: Dict[str, Dict] = {}
        self.total_value = initial_capital # Current total value (cash + holdings)
        self.peak_value = initial_capital # Track peak portfolio value for drawdown calculation
        self.trade_count = 0
        self.halt_trading_flag = False # Flag to halt trading due to drawdown
        log.info(f"PortfolioManager initialized with capital: ${self.cash:.2f}")

    def update_position(self, symbol: str, quantity_change: float, price: float, timestamp: pd.Timestamp):
        """
        Updates cash and position quantity after a trade execution.

        Args:
            symbol (str): The symbol traded.
            quantity_change (float): Change in quantity (+ for buy, - for sell).
            price (float): Execution price per unit.
            timestamp (pd.Timestamp): Time of the trade execution.
        """
        trade_value = abs(quantity_change) * price
        self.trade_count += 1

        if quantity_change > 0: # Buy order
            self.cash -= trade_value
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': quantity_change, 'entry_price': price, 'entry_time': timestamp}
                log.info(f"Opened position: BUY {quantity_change:.8f} {symbol} @ {price:.4f}")
            else:
                # Update average entry price if adding to existing position (optional, simpler to just update quantity)
                # current_qty = self.positions[symbol]['quantity']
                # current_entry = self.positions[symbol]['entry_price']
                # new_avg_price = ((current_qty * current_entry) + (quantity_change * price)) / (current_qty + quantity_change)
                # self.positions[symbol]['entry_price'] = new_avg_price
                self.positions[symbol]['quantity'] += quantity_change
                log.info(f"Increased position: BUY {quantity_change:.8f} {symbol} @ {price:.4f}. New Qty: {self.positions[symbol]['quantity']:.8f}")

        elif quantity_change < 0: # Sell order
            sell_quantity = abs(quantity_change)
            self.cash += trade_value
            if symbol in self.positions:
                # Calculate realized PnL for this sell portion
                entry_price = self.positions[symbol]['entry_price']
                pnl = (price - entry_price) * sell_quantity
                log.info(f"Closed/Reduced position: SELL {sell_quantity:.8f} {symbol} @ {price:.4f}. Realized PnL: ${pnl:.2f}")

                self.positions[symbol]['quantity'] -= sell_quantity
                if self.positions[symbol]['quantity'] <= 1e-9: # Use tolerance for float comparison
                    log.info(f"Position closed for {symbol}.")
                    del self.positions[symbol]
                else:
                     log.info(f"Remaining Qty for {symbol}: {self.positions[symbol]['quantity']:.8f}")
            else:
                # This case (selling without a position) should ideally be prevented by the trader logic
                log.warning(f"Attempted to SELL {sell_quantity:.8f} {symbol} but no position was held.")
                # Adjust cash anyway if the sell somehow executed (e.g., in live mode due to external factors)
                # Or handle as an error depending on desired strictness

        log.debug(f"Portfolio updated: Cash: ${self.cash:.2f}, Positions: {len(self.positions)}")


    def get_position(self, symbol: str) -> Optional[Dict]:
        """Returns current position details for a symbol, or None if not held."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Returns all currently open positions."""
        return self.positions.copy() # Return a copy

    def get_available_capital(self) -> float:
        """Returns the current cash balance."""
        # Could potentially reserve some cash or consider margin requirements later
        return self.cash

    def calculate_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculates the current total portfolio value (cash + value of holdings).
        Updates self.total_value and self.peak_value.

        Args:
            current_prices (Dict[str, float]): Dictionary mapping symbols to their current market price.

        Returns:
            float: The calculated total portfolio value.
        """
        holdings_value = 0.0
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol)
            if current_price is not None:
                holdings_value += position['quantity'] * current_price
            else:
                # Handle missing price - use entry price? Log warning? Exclude?
                log.warning(f"Missing current price for holding {symbol}. Using entry price for value calculation.")
                holdings_value += position['quantity'] * position['entry_price']

        self.total_value = self.cash + holdings_value
        self.peak_value = max(self.peak_value, self.total_value) # Update peak value
        log.debug(f"Portfolio value calculated: Total=${self.total_value:.2f}, Peak=${self.peak_value:.2f}, Cash=${self.cash:.2f}, Holdings=${holdings_value:.2f}")
        return self.total_value

    def check_drawdown_and_halt(self, drawdown_pct: float = config.PORTFOLIO_DRAWDOWN_PCT) -> bool:
        """
        Checks if the portfolio has hit the maximum drawdown limit.
        Sets the halt_trading_flag if drawdown is breached.

        Args:
            drawdown_pct (float): The maximum allowed drawdown percentage (e.g., 0.15 for 15%).

        Returns:
            bool: True if drawdown limit is breached, False otherwise.
        """
        if self.halt_trading_flag: # Already halted
            return True

        if self.peak_value <= 0: # Avoid division by zero if starting capital was 0 or negative
            return False

        current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        drawdown_limit_breached = current_drawdown >= drawdown_pct

        if drawdown_limit_breached:
            log.critical(f"MAXIMUM DRAWDOWN LIMIT REACHED! Current Value: ${self.total_value:.2f}, Peak Value: ${self.peak_value:.2f}, Drawdown: {current_drawdown:.2%}. Halting all new trading.")
            self.halt_trading_flag = True
            return True
        else:
            log.debug(f"Current Drawdown: {current_drawdown:.2%}. Limit: {drawdown_pct:.2%}")
            return False

    def reset_halt_flag(self):
        """Manually resets the halt trading flag (e.g., via dashboard)."""
        if self.halt_trading_flag:
            log.warning("Resetting portfolio halt_trading_flag manually.")
            self.halt_trading_flag = False

    def get_trade_size_usd(self, capital_percentage: float = config.TRADE_CAPITAL_PERCENTAGE) -> float:
        """
        Calculates the target USD value for a new trade based on a percentage of available capital.

        Args:
            capital_percentage (float): The percentage of available capital to allocate (e.g., 0.01 for 1%).

        Returns:
            float: The calculated USD value for the trade.
        """
        if self.halt_trading_flag:
            return 0.0 # No new trades if halted

        available = self.get_available_capital()
        trade_size = available * capital_percentage
        log.debug(f"Calculated trade size: {capital_percentage:.2%} of ${available:.2f} = ${trade_size:.2f}")
        return max(0.0, trade_size) # Ensure non-negative

    def get_trade_quantity(self, symbol: str, current_price: float, capital_percentage: float = config.TRADE_CAPITAL_PERCENTAGE) -> float:
        """
        Calculates the quantity of a symbol to trade based on a percentage of capital.

        Args:
            symbol (str): The symbol to trade.
            current_price (float): The current market price of the symbol.
            capital_percentage (float): The percentage of available capital to allocate.

        Returns:
            float: The quantity to trade (can be 0). Returns 0 if price is invalid or trade size is too small.
        """
        if current_price <= 0:
            log.warning(f"Cannot calculate quantity for {symbol}: Invalid current price ({current_price}).")
            return 0.0

        trade_usd = self.get_trade_size_usd(capital_percentage)
        if trade_usd <= 0:
            return 0.0

        # TODO: Consider Binance minimum order size requirements (LOT_SIZE filter)
        # This calculation might result in a quantity too small to trade.
        # Need to fetch exchange info filters for the symbol.
        quantity = trade_usd / current_price
        log.debug(f"Calculated quantity for ${trade_usd:.2f} trade of {symbol} @ {current_price:.4f}: {quantity:.8f}")

        # --- Placeholder for LOT_SIZE adjustment ---
        # min_qty, step_size = get_lot_size_filter(symbol) # Function to get from exchange info
        # if quantity < min_qty:
        #     log.warning(f"Calculated quantity {quantity} for {symbol} is below minimum lot size {min_qty}. Setting quantity to 0.")
        #     return 0.0
        # quantity = floor(quantity / step_size) * step_size # Adjust to step size
        # ------------------------------------------

        return quantity


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Portfolio Manager ---")
    pm = PortfolioManager(initial_capital=10000.0)
    print(f"Initial Cash: ${pm.get_available_capital():.2f}")

    # Simulate a BUY
    symbol = 'TESTBTC'
    buy_price = 50000.0
    buy_time = pd.Timestamp.now(tz=pytz.utc)
    # Calculate quantity for 2% of capital
    buy_qty = pm.get_trade_quantity(symbol, buy_price, capital_percentage=0.02)
    if buy_qty > 0:
        print(f"\nSimulating BUY {buy_qty:.8f} {symbol} @ {buy_price}")
        pm.update_position(symbol, buy_qty, buy_price, buy_time)
        print(f"Cash after BUY: ${pm.get_available_capital():.2f}")
        print(f"Position: {pm.get_position(symbol)}")
    else:
        print("\nCould not execute BUY (quantity is 0).")


    # Simulate price update and value calculation
    current_prices = {symbol: 51000.0, 'OTHER': 100.0}
    print(f"\nCalculating value with current prices: {current_prices}")
    total_val = pm.calculate_total_value(current_prices)
    print(f"Total Portfolio Value: ${total_val:.2f}")

    # Simulate a SELL
    sell_price = 50500.0 # Sell for a small loss
    sell_time = pd.Timestamp.now(tz=pytz.utc)
    position_to_sell = pm.get_position(symbol)
    if position_to_sell:
        sell_qty = position_to_sell['quantity'] # Sell entire position
        print(f"\nSimulating SELL {sell_qty:.8f} {symbol} @ {sell_price}")
        pm.update_position(symbol, -sell_qty, sell_price, sell_time)
        print(f"Cash after SELL: ${pm.get_available_capital():.2f}")
        print(f"Position: {pm.get_position(symbol)}") # Should be None
    else:
        print("\nNo position to sell.")

    # Check value again
    total_val_after_sell = pm.calculate_total_value(current_prices)
    print(f"\nTotal Portfolio Value after SELL: ${total_val_after_sell:.2f}")

    # Simulate Drawdown
    print("\nSimulating Drawdown Check...")
    pm.total_value = 8000 # Manually set value below initial peak of 10000
    is_halted = pm.check_drawdown_and_halt(drawdown_pct=0.15) # 15% drawdown limit
    print(f"Drawdown limit breached? {is_halted}")
    print(f"Trading halted flag: {pm.halt_trading_flag}")
    # Try getting trade size while halted
    print(f"Attempting to get trade size while halted: ${pm.get_trade_size_usd():.2f}")


    print("\n--- Test Complete ---")


