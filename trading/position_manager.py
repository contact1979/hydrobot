"""Position and P&L tracking module.

Maintains real-time position state and calculates P&L metrics
for risk management decisions.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from utilities.logger_setup import get_logger, log_position_update

logger = get_logger(__name__)

@dataclass
class Position:
    """Position state data class."""
    symbol: str
    size: float
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    timestamp: datetime
    trades: List[Dict[str, Any]]

class PositionManager:
    def __init__(self, config: Dict[str, Any]):
        """Initialize position manager.
        
        Args:
            config: Position management configuration
        """
        self.config = config
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        self._trade_history: Dict[str, List[Dict]] = {}
        
        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_volume = 0.0
        self._last_reset = datetime.utcnow()
    
    def initialize_symbol(self, symbol: str) -> None:
        """Initialize tracking for new symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        self._positions[symbol] = Position(
            symbol=symbol,
            size=0.0,
            entry_price=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            timestamp=datetime.utcnow(),
            trades=[]
        )
        self._trade_history[symbol] = []
    
    def _calculate_pnl(self,
                     symbol: str,
                     current_price: float
                     ) -> Dict[str, float]:
        """Calculate position P&L metrics.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        
        Returns:
            Dict containing P&L metrics
        """
        position = self._positions[symbol]
        
        if position.size == 0:
            return {
                'unrealized_pnl': 0.0,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.realized_pnl
            }
        
        # Calculate unrealized P&L
        unrealized_pnl = (
            position.size * (current_price - position.entry_price)
        )
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': unrealized_pnl + position.realized_pnl
        }
    
    def update_position(self,
                      symbol: str,
                      trade: Dict[str, Any],
                      current_price: float
                      ) -> Position:
        """Update position with new trade.
        
        Args:
            symbol: Trading pair symbol
            trade: Trade execution details
            current_price: Current market price
        
        Returns:
            Updated position state
        """
        try:
            if symbol not in self._positions:
                self.initialize_symbol(symbol)
            
            position = self._positions[symbol]
            
            # Extract trade details
            size = trade['size']
            price = trade['price']
            side = trade['side']
            
            # Calculate trade P&L if closing
            old_size = position.size
            new_size = old_size + (size if side == 'buy' else -size)
            
            if abs(new_size) < abs(old_size):  # Closing trade
                # Calculate realized P&L
                realized_pnl = (
                    abs(size) * (price - position.entry_price)
                    if side == 'sell' else
                    abs(size) * (position.entry_price - price)
                )
                position.realized_pnl += realized_pnl
            
            # Update position
            if new_size != 0:
                # Update entry price with weighted average
                position.entry_price = (
                    (old_size * position.entry_price + size * price) /
                    (old_size + (size if side == 'buy' else -size))
                )
            else:
                position.entry_price = 0.0
            
            position.size = new_size
            
            # Update P&L
            pnl = self._calculate_pnl(symbol, current_price)
            position.unrealized_pnl = pnl['unrealized_pnl']
            position.total_pnl = pnl['total_pnl']
            position.timestamp = datetime.utcnow()
            
            # Store trade
            position.trades.append(trade)
            self._trade_history[symbol].append(trade)
            
            # Update daily stats
            self._daily_trades += 1
            self._daily_volume += size * price
            self._daily_pnl = sum(p.total_pnl for p in self._positions.values())
            
            # Log position update
            log_position_update(logger, symbol, new_size)
            
            return position
            
        except Exception as e:
            logger.error("position_update_error",
                        error=str(e),
                        symbol=symbol,
                        trade=trade)
            return self._positions.get(symbol)
    
    def mark_to_market(self,
                     symbol: str,
                     current_price: float
                     ) -> Position:
        """Update position with current market price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        
        Returns:
            Updated position state
        """
        try:
            if symbol not in self._positions:
                self.initialize_symbol(symbol)
            
            position = self._positions[symbol]
            
            # Update P&L
            pnl = self._calculate_pnl(symbol, current_price)
            position.unrealized_pnl = pnl['unrealized_pnl']
            position.total_pnl = pnl['total_pnl']
            position.timestamp = datetime.utcnow()
            
            return position
            
        except Exception as e:
            logger.error("mark_to_market_error",
                        error=str(e),
                        symbol=symbol,
                        price=current_price)
            return self._positions.get(symbol)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Current position state
        """
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions.
        
        Returns:
            Dict of all positions
        """
        return self._positions.copy()
    
    def get_trade_history(self,
                        symbol: Optional[str] = None,
                        limit: int = 100
                        ) -> List[Dict[str, Any]]:
        """Get recent trade history.
        
        Args:
            symbol: Optional symbol filter
            limit: Number of trades to return
        
        Returns:
            List of recent trades
        """
        if symbol:
            trades = self._trade_history.get(symbol, [])
        else:
            trades = [
                trade for trades in self._trade_history.values()
                for trade in trades
            ]
        return sorted(trades, key=lambda x: x['timestamp'])[-limit:]
    
    def get_daily_stats(self) -> Dict[str, float]:
        """Get daily trading statistics.
        
        Returns:
            Dict of daily metrics
        """
        return {
            'pnl': self._daily_pnl,
            'trades': self._daily_trades,
            'volume': self._daily_volume
        }
    
    def reset_daily_stats(self) -> None:
        """Reset daily trading statistics."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_volume = 0.0
        self._last_reset = datetime.utcnow()
        
        logger.info("daily_stats_reset",
                   timestamp=self._last_reset)