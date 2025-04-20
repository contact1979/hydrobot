"""Risk management and circuit breaker controller.

Enforces position limits, monitors drawdown, and implements
circuit breaker logic to halt trading when risk limits are breached.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from .position_manager import PositionManager, Position
from utilities.logger_setup import get_logger
from utilities.metrics import (
    RISK_EVENTS,
    DRAWDOWN,
    PNL,
    POSITION_VALUE
)

logger = get_logger(__name__)

class RiskBreachType(Enum):
    """Types of risk limit breaches."""
    POSITION_LIMIT = "position_limit"
    DRAWDOWN = "drawdown"
    ERROR_THRESHOLD = "error_threshold"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    VOLATILITY = "volatility"
    CUSTOM = "custom"

@dataclass
class RiskBreach:
    """Risk limit breach event."""
    type: RiskBreachType
    symbol: str
    timestamp: datetime
    threshold: float
    current_value: float
    metadata: Dict[str, Any]

class CircuitBreaker:
    """Circuit breaker state management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.cooldown_seconds = config.get('circuit_breaker_cooldown', 300)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.05)
        
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.breaches: List[RiskBreach] = []
        self.consecutive_losses = 0
        self.peak_equity = 0.0
    
    def update_metrics(self, pnl: float, equity: float) -> None:
        """Update tracking metrics.
        
        Args:
            pnl: Latest trade P&L
            equity: Current equity
        """
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Track peak equity
        self.peak_equity = max(self.peak_equity, equity)
    
    def should_trigger(self, equity: float) -> Optional[RiskBreach]:
        """Check if circuit breaker should trigger.
        
        Args:
            equity: Current equity
        
        Returns:
            Risk breach if triggered
        """
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return RiskBreach(
                type=RiskBreachType.CONSECUTIVE_LOSSES,
                symbol='all',
                timestamp=datetime.utcnow(),
                threshold=self.max_consecutive_losses,
                current_value=self.consecutive_losses,
                metadata={'consecutive_losses': self.consecutive_losses}
            )
        
        # Check drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.max_drawdown_pct:
                return RiskBreach(
                    type=RiskBreachType.DRAWDOWN,
                    symbol='all',
                    timestamp=datetime.utcnow(),
                    threshold=self.max_drawdown_pct,
                    current_value=drawdown,
                    metadata={
                        'peak_equity': self.peak_equity,
                        'current_equity': equity
                    }
                )
        
        return None
    
    def trigger(self, breach: RiskBreach) -> None:
        """Trigger circuit breaker.
        
        Args:
            breach: Risk breach event
        """
        self.is_triggered = True
        self.trigger_time = datetime.utcnow()
        self.breaches.append(breach)
        
        # Record metric
        RISK_EVENTS.labels(
            type=breach.type.value,
            symbol=breach.symbol
        ).inc()
        
        logger.warning("circuit_breaker_triggered",
                      breach_type=breach.type.value,
                      symbol=breach.symbol,
                      threshold=breach.threshold,
                      value=breach.current_value)
    
    def check_reset(self) -> bool:
        """Check if cooldown period has passed.
        
        Returns:
            True if circuit breaker should reset
        """
        if not self.is_triggered or not self.trigger_time:
            return False
        
        elapsed = datetime.utcnow() - self.trigger_time
        return elapsed.total_seconds() >= self.cooldown_seconds
    
    def reset(self) -> None:
        """Reset circuit breaker state."""
        self.is_triggered = False
        self.trigger_time = None
        self.consecutive_losses = 0
        logger.info("circuit_breaker_reset",
                   cooldown_seconds=self.cooldown_seconds)

class RiskController:
    def __init__(self, config: Dict[str, Any], position_manager: PositionManager):
        """Initialize risk controller.
        
        Args:
            config: Risk management configuration
            position_manager: Position manager instance
        """
        self.config = config
        self.position_manager = position_manager
        
        # Risk limits
        self.max_position_size = config.get('max_position_size', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.05)  # 5%
        self.daily_loss_limit = config.get('daily_loss_limit', 0.02)  # 2%
        self.error_threshold = config.get('error_threshold', 3)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(config)
        
        # Error tracking
        self._error_count = 0
        self._error_window: List[datetime] = []
        self._error_window_seconds = 300  # 5 minutes
        
        # Start monitoring task
        self._running = True
        asyncio.create_task(self._monitor_risk_limits())
    
    async def _monitor_risk_limits(self) -> None:
        """Periodic risk limit monitoring."""
        while self._running:
            try:
                # Check risk limits
                breach = self.check_risk_limits()
                if breach:
                    self.process_risk_breach(breach)
                
                # Check circuit breaker reset
                if self.circuit_breaker.is_triggered:
                    if self.circuit_breaker.check_reset():
                        self.circuit_breaker.reset()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error("risk_monitor_error", error=str(e))
                await asyncio.sleep(5)  # Back off on error
    
    def _check_position_limits(self, position: Position) -> Optional[RiskBreach]:
        """Check if position exceeds limits.
        
        Args:
            position: Current position
        
        Returns:
            Risk breach if limit exceeded
        """
        if abs(position.size) > self.max_position_size:
            return RiskBreach(
                type=RiskBreachType.POSITION_LIMIT,
                symbol=position.symbol,
                timestamp=datetime.utcnow(),
                threshold=self.max_position_size,
                current_value=abs(position.size),
                metadata={'position': position.__dict__}
            )
        return None
    
    def _check_drawdown(self, stats: Dict[str, float]) -> Optional[RiskBreach]:
        """Check if drawdown exceeds limits.
        
        Args:
            stats: Daily trading statistics
        
        Returns:
            Risk breach if limit exceeded
        """
        daily_pnl = stats['pnl']
        if daily_pnl < -self.daily_loss_limit:
            return RiskBreach(
                type=RiskBreachType.DRAWDOWN,
                symbol='all',  # Applies to all symbols
                timestamp=datetime.utcnow(),
                threshold=-self.daily_loss_limit,
                current_value=daily_pnl,
                metadata={'daily_stats': stats}
            )
        return None
    
    def _check_error_threshold(self) -> Optional[RiskBreach]:
        """Check if error count exceeds threshold.
        
        Returns:
            Risk breach if threshold exceeded
        """
        # Remove old errors
        cutoff = datetime.utcnow() - timedelta(seconds=self._error_window_seconds)
        self._error_window = [t for t in self._error_window if t > cutoff]
        
        if len(self._error_window) >= self.error_threshold:
            return RiskBreach(
                type=RiskBreachType.ERROR_THRESHOLD,
                symbol='all',
                timestamp=datetime.utcnow(),
                threshold=self.error_threshold,
                current_value=len(self._error_window),
                metadata={'error_times': [t.isoformat() for t in self._error_window]}
            )
        return None
    
    def check_risk_limits(self) -> Optional[RiskBreach]:
        """Check all risk limits.
        
        Returns:
            Risk breach if any limit exceeded
        """
        try:
            # Get current state
            positions = self.position_manager.get_all_positions()
            daily_stats = self.position_manager.get_daily_stats()
            
            # Update circuit breaker metrics
            self.circuit_breaker.update_metrics(
                daily_stats['pnl'],
                sum(p.total_pnl for p in positions.values())
            )
            
            # Check circuit breaker conditions
            if breach := self.circuit_breaker.should_trigger(
                sum(p.total_pnl for p in positions.values())
            ):
                return breach
            
            # Check position limits
            for position in positions.values():
                # Update position metrics
                POSITION_VALUE.labels(symbol=position.symbol).set(
                    position.size * position.entry_price
                )
                PNL.labels(symbol=position.symbol, type='realized').set(
                    position.realized_pnl
                )
                PNL.labels(symbol=position.symbol, type='unrealized').set(
                    position.unrealized_pnl
                )
                DRAWDOWN.labels(symbol=position.symbol).set(
                    max(0, position.entry_price - position.unrealized_pnl)
                    / position.entry_price if position.entry_price > 0 else 0
                )
                
                if breach := self._check_position_limits(position):
                    return breach
            
            # Check drawdown
            if breach := self._check_drawdown(daily_stats):
                return breach
            
            # Check error threshold
            if breach := self._check_error_threshold():
                return breach
            
            return None
            
        except Exception as e:
            logger.error("risk_check_error", error=str(e))
            return None
    
    def can_trade(self, symbol: str) -> bool:
        """Check if trading is allowed.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            True if trading is allowed
        """
        return not self.circuit_breaker.is_triggered
    
    def record_error(self) -> None:
        """Record trading error occurrence."""
        self._error_window.append(datetime.utcnow())
        
        # Record metric
        RISK_EVENTS.labels(
            type='error',
            symbol='all'
        ).inc()
    
    def process_risk_breach(self, breach: RiskBreach) -> None:
        """Handle risk limit breach.
        
        Args:
            breach: Risk breach event
        """
        # Trigger circuit breaker
        self.circuit_breaker.trigger(breach)
        
        # Record metrics
        RISK_EVENTS.labels(
            type=breach.type.value,
            symbol=breach.symbol
        ).inc()
        
        logger.error("risk_limit_breached",
                    type=breach.type.value,
                    symbol=breach.symbol,
                    threshold=breach.threshold,
                    value=breach.current_value)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk management status.
        
        Returns:
            Dict containing risk status
        """
        return {
            'circuit_breaker_active': self.circuit_breaker.is_triggered,
            'trigger_time': self.circuit_breaker.trigger_time,
            'consecutive_losses': self.circuit_breaker.consecutive_losses,
            'recent_breaches': [b.__dict__ for b in self.circuit_breaker.breaches[-5:]],
            'error_count': len(self._error_window),
            'daily_stats': self.position_manager.get_daily_stats()
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        self._running = False