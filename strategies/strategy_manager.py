"""Strategy manager for coordinating multiple trading strategies.

Manages strategy lifecycle and coordinates strategy selection based on
market regime and conditions.
"""
from typing import Dict, Any, List, Type, Optional
from .base_strategy import Strategy, Signal
from .scalping_strategy import ScalpingStrategy
from ml_models.regime_model import MarketRegime
from utilities.logger_setup import get_logger

logger = get_logger(__name__)

class StrategyManager:
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy manager.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.active_strategies: Dict[str, Strategy] = {}
        
        # Track recent regimes for smooth transitions
        self._regime_history: Dict[str, List[MarketRegime]] = {}
        self._regime_window = 5  # Number of regimes to track
        
        # Strategy mapping - using ScalpingStrategy for all regimes
        # but with dynamic parameter adjustment
        self.strategy_configs: Dict[MarketRegime, Type[Strategy]] = {
            MarketRegime.RANGING_LOW_VOL: ScalpingStrategy,
            MarketRegime.RANGING_HIGH_VOL: ScalpingStrategy,
            MarketRegime.TRENDING_UP: ScalpingStrategy,
            MarketRegime.TRENDING_DOWN: ScalpingStrategy,
            MarketRegime.VOLATILE: ScalpingStrategy,
            MarketRegime.UNKNOWN: ScalpingStrategy
        }

    async def initialize(self) -> bool:
        """Initialize strategy manager and default strategies.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize default strategy for each trading pair
            for symbol in self.config.get('trading_pairs', []):
                strategy = ScalpingStrategy(self.config)
                if await strategy.initialize():
                    self.active_strategies[symbol] = strategy
                    self._regime_history[symbol] = []
                else:
                    logger.error("strategy_init_failed", symbol=symbol)
                    return False
            
            return True
            
        except Exception as e:
            logger.error("strategy_manager_init_error", error=str(e))
            return False
    
    def _select_strategy(self, regime: MarketRegime) -> Type[Strategy]:
        """Select appropriate strategy for market regime.
        
        Args:
            regime: Current market regime
        
        Returns:
            Strategy class to use
        """
        return self.strategy_configs.get(regime, ScalpingStrategy)
    
    def _is_regime_stable(self, symbol: str, current_regime: MarketRegime) -> bool:
        """Check if market regime has been stable.
        
        Args:
            symbol: Trading pair symbol
            current_regime: Current market regime
        
        Returns:
            True if regime is considered stable
        """
        history = self._regime_history.get(symbol, [])
        if len(history) < 3:  # Need minimum history
            return True
            
        # Check if regime has been consistent
        return all(r == current_regime for r in history[-3:])
    
    async def update_strategy(self,
                           symbol: str,
                           regime: MarketRegime,
                           volatility: float
                           ) -> None:
        """Update or switch strategy based on market conditions.
        
        Args:
            symbol: Trading pair symbol
            regime: Current market regime
            volatility: Current volatility level
        """
        try:
            # Update regime history
            history = self._regime_history.get(symbol, [])
            history.append(regime)
            self._regime_history[symbol] = history[-self._regime_window:]
            
            current_strategy = self.active_strategies.get(symbol)
            if not current_strategy:
                return
            
            # Only consider switching if regime is stable
            if self._is_regime_stable(symbol, regime):
                optimal_strategy_class = self._select_strategy(regime)
                if not isinstance(current_strategy, optimal_strategy_class):
                    # Create and initialize new strategy
                    new_strategy = optimal_strategy_class(self.config)
                    if await new_strategy.initialize():
                        # Stop old strategy and switch
                        await current_strategy.stop()
                        self.active_strategies[symbol] = new_strategy
                        await new_strategy.start()
                        
                        logger.info("strategy_switched",
                                  symbol=symbol,
                                  old_strategy=type(current_strategy).__name__,
                                  new_strategy=type(new_strategy).__name__,
                                  regime=regime.name)
            
            # Update strategy parameters
            current_strategy.update_parameters(regime, volatility)
            
        except Exception as e:
            logger.error("strategy_update_error",
                        error=str(e),
                        symbol=symbol,
                        regime=regime.name)
    
    async def process_market_update(self,
                                 symbol: str,
                                 market_data: Dict[str, Any],
                                 model_predictions: Dict[str, Any]
                                 ) -> Optional[Signal]:
        """Process market update with active strategy.
        
        Args:
            symbol: Trading pair symbol
            market_data: Current market state
            model_predictions: ML model predictions
        
        Returns:
            Optional trading signal
        """
        try:
            strategy = self.active_strategies.get(symbol)
            if not strategy:
                return None
            
            return await strategy.on_market_update(
                symbol, market_data, model_predictions)
            
        except Exception as e:
            logger.error("market_update_error",
                        error=str(e),
                        symbol=symbol)
            return None
    
    async def process_trade_update(self,
                                symbol: str,
                                trade_update: Dict[str, Any]
                                ) -> None:
        """Process trade update with active strategy.
        
        Args:
            symbol: Trading pair symbol
            trade_update: Trade execution details
        """
        try:
            strategy = self.active_strategies.get(symbol)
            if strategy:
                await strategy.on_trade_update(trade_update)
            
        except Exception as e:
            logger.error("trade_update_error",
                        error=str(e),
                        symbol=symbol)
    
    async def process_order_update(self,
                                symbol: str,
                                order_update: Dict[str, Any]
                                ) -> None:
        """Process order update with active strategy.
        
        Args:
            symbol: Trading pair symbol
            order_update: Order status details
        """
        try:
            strategy = self.active_strategies.get(symbol)
            if strategy:
                await strategy.on_order_update(order_update)
            
        except Exception as e:
            logger.error("order_update_error",
                        error=str(e),
                        symbol=symbol)
    
    async def process_position_update(self,
                                   symbol: str,
                                   position_update: Dict[str, Any]
                                   ) -> None:
        """Process position update with active strategy.
        
        Args:
            symbol: Trading pair symbol
            position_update: Position details
        """
        try:
            strategy = self.active_strategies.get(symbol)
            if strategy:
                await strategy.on_position_update(position_update)
            
        except Exception as e:
            logger.error("position_update_error",
                        error=str(e),
                        symbol=symbol)
    
    async def start_all(self) -> None:
        """Start all managed strategies."""
        for symbol, strategy in self.active_strategies.items():
            await strategy.start()
            logger.info("strategy_started",
                       symbol=symbol,
                       strategy=type(strategy).__name__)
    
    async def stop_all(self) -> None:
        """Stop all managed strategies."""
        for symbol, strategy in self.active_strategies.items():
            await strategy.stop()
            logger.info("strategy_stopped",
                       symbol=symbol,
                       strategy=type(strategy).__name__)