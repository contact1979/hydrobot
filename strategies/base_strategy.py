"""Base strategy interface for HFT scalping bot.

Defines the contract that all trading strategies must implement,
including strategy lifecycle and signal generation methods.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from ml_models.regime_model import MarketRegime
from utilities.logger_setup import get_logger

logger = get_logger(__name__)

@dataclass
class Signal:
    """Trading signal data class."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    type: str  # 'market' or 'limit'
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    confidence: float
    metadata: Dict[str, Any]  # Additional signal context

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.is_active = False
        self._last_signal: Optional[Signal] = None
        self._signal_history: List[Signal] = []
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize strategy resources.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def on_market_update(self,
                             symbol: str,
                             market_data: Dict[str, Any],
                             model_predictions: Dict[str, Any]
                             ) -> Optional[Signal]:
        """Process market update and generate trading signal.
        
        Args:
            symbol: Trading pair symbol
            market_data: Current market state
            model_predictions: ML model predictions
        
        Returns:
            Optional trading signal
        """
        pass
    
    @abstractmethod
    async def on_trade_update(self,
                            trade_update: Dict[str, Any]
                            ) -> None:
        """Process trade execution update.
        
        Args:
            trade_update: Trade execution details
        """
        pass
    
    @abstractmethod
    async def on_order_update(self,
                           order_update: Dict[str, Any]
                           ) -> None:
        """Process order status update.
        
        Args:
            order_update: Order status details
        """
        pass
    
    @abstractmethod
    async def on_position_update(self,
                              position_update: Dict[str, Any]
                              ) -> None:
        """Process position update.
        
        Args:
            position_update: Current position details
        """
        pass
    
    @abstractmethod
    def update_parameters(self,
                        market_regime: MarketRegime,
                        volatility: float
                        ) -> None:
        """Update strategy parameters based on market conditions.
        
        Args:
            market_regime: Current market regime
            volatility: Current volatility level
        """
        pass
    
    def _validate_signal(self, signal: Signal) -> bool:
        """Validate trading signal meets requirements.
        
        Args:
            signal: Trading signal to validate
        
        Returns:
            True if signal is valid
        """
        try:
            # Basic validation
            if signal.size <= 0 or signal.price <= 0:
                return False
            
            # Validate side
            if signal.side not in ['buy', 'sell']:
                return False
            
            # Validate order type
            if signal.type not in ['market', 'limit']:
                return False
            
            # Validate time in force
            if signal.time_in_force not in ['GTC', 'IOC', 'FOK']:
                return False
            
            # Validate confidence
            if not 0 <= signal.confidence <= 1:
                return False
            
            return True
            
        except Exception as e:
            logger.error("signal_validation_error",
                        error=str(e),
                        signal=signal.__dict__)
            return False
    
    def add_signal_to_history(self, signal: Signal) -> None:
        """Add validated signal to history.
        
        Args:
            signal: Trading signal to store
        """
        if self._validate_signal(signal):
            self._signal_history.append(signal)
            self._last_signal = signal
            
            # Limit history size
            if len(self._signal_history) > 1000:
                self._signal_history = self._signal_history[-1000:]
    
    def get_last_signal(self) -> Optional[Signal]:
        """Get most recent trading signal.
        
        Returns:
            Last generated signal or None
        """
        return self._last_signal
    
    def get_signal_history(self) -> List[Signal]:
        """Get recent signal history.
        
        Returns:
            List of recent signals
        """
        return self._signal_history.copy()
    
    async def start(self) -> None:
        """Start strategy execution."""
        self.is_active = True
        logger.info("strategy_started",
                   config=self.config)
    
    async def stop(self) -> None:
        """Stop strategy execution."""
        self.is_active = False
        logger.info("strategy_stopped")