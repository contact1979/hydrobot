"""HFT scalping strategy implementation.

Uses market microstructure signals, ML predictions, and order book
imbalance to identify and execute high-frequency scalping opportunities.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
from .base_strategy import Strategy, Signal
from ml_models.regime_model import MarketRegime
from utilities.logger_setup import get_logger
from utilities.metrics import (
    REGIME_TRANSITIONS, VOLATILITY_SCALE, POSITION_SCALING,
    CONFIDENCE_THRESHOLDS, DAILY_METRICS, DRAWDOWN_METRICS
)

logger = get_logger(__name__)

class ScalpingStrategy(Strategy):
    def __init__(self, config: Dict[str, Any]):
        """Initialize scalping strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Core parameters from config
        self.min_spread = config.get('min_spread', 0.001)
        self.max_position_size = config.get('max_position_size', 0.005)  # Updated from config
        self.base_order_size = config.get('base_order_size', 0.001)
        self.max_slippage = config.get('max_slippage', 0.002)
        
        # Dynamic thresholds
        self.min_imbalance = 0.2
        self.confidence_threshold = 0.7
        self.volatility_scale = 1.0
        
        # Risk limits
        self.max_drawdown = config.get('max_drawdown', 0.05)
        self.daily_loss_limit = config.get('daily_loss_limit', 0.02)
        
        # State tracking
        self.symbol = config.get('symbol', 'default_symbol')  # Initialize symbol from config
        self._positions: Dict[str, float] = {}
        self._recent_trades: List[Dict] = []
        self._last_signals: Dict[str, Signal] = {}
        self._daily_pnl = 0.0
        self._session_peak_pnl = 0.0
    
    async def initialize(self) -> bool:
        """Initialize strategy state.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize empty positions
            for symbol in self.config.get('trading_pairs', []):
                self._positions[symbol] = 0.0
            return True
            
        except Exception as e:
            logger.error("strategy_init_error", error=str(e))
            return False
    
    def _calculate_trade_size(self,
                           symbol: str,
                           current_price: float,
                           confidence: float
                           ) -> float:
        """Calculate trade size based on position and confidence.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            confidence: Signal confidence score
        
        Returns:
            Trade size in base currency units
        """
        # Scale size by confidence and available capacity
        current_position = abs(self._positions.get(symbol, 0))
        remaining_capacity = self.max_position_size - current_position
        
        if remaining_capacity <= 0:
            return 0
        
        # Dynamic size based on regime volatility
        adjusted_size = self.base_order_size * self.volatility_scale
        
        # Scale by confidence
        size = adjusted_size * confidence
        
        # Apply daily loss limit scaling
        if self._daily_pnl < 0:
            loss_scale = max(0.5, 1 + (self._daily_pnl / self.daily_loss_limit))
            size *= loss_scale
        
        # Limit to remaining capacity
        return min(size, remaining_capacity)
    
    def _should_open_position(self,
                           imbalance: float,
                           spread: float,
                           prediction: float,
                           confidence: float,
                           regime: MarketRegime
                           ) -> bool:
        """Check if conditions are right to open new position.
        
        Args:
            imbalance: Order book imbalance
            spread: Current bid-ask spread
            prediction: Price prediction
            confidence: Model confidence
            regime: Current market regime
        
        Returns:
            True if should open position
        """
        # Check basic conditions
        if confidence < self.confidence_threshold:
            return False
        
        if spread < self.min_spread:
            return False
        
        if abs(imbalance) < self.min_imbalance:
            return False
        
        # Don't trade if near daily loss limit
        if self._daily_pnl <= -self.daily_loss_limit:
            return False
        
        # Check drawdown limit
        current_drawdown = (self._session_peak_pnl - self._daily_pnl) / max(1.0, self._session_peak_pnl)
        if current_drawdown >= self.max_drawdown:
            return False
        
        # Regime-specific checks
        if regime == MarketRegime.VOLATILE:
            # Only trade with very high confidence in volatile markets
            return confidence > 0.8 and abs(imbalance) > self.min_imbalance * 1.5
        
        elif regime == MarketRegime.RANGING_LOW_VOL:
            # More opportunities in low vol markets
            return abs(imbalance) > self.min_imbalance * 0.8
        
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Ensure prediction aligns with trend
            trend_direction = 1 if regime == MarketRegime.TRENDING_UP else -1
            pred_direction = 1 if prediction > 0 else -1
            return trend_direction == pred_direction
        
        return True
    
    async def on_market_update(self,
                             symbol: str,
                             market_data: Dict[str, Any],
                             model_predictions: Dict[str, Any]
                             ) -> Optional[Signal]:
        """Process market update and generate trading signals.
        
        Args:
            symbol: Trading pair symbol
            market_data: Current market state
            model_predictions: ML model predictions
        
        Returns:
            Optional trading signal
        """
        try:
            if not self.is_active:
                return None
            
            # Extract relevant data
            orderbook = market_data['orderbook']
            imbalance = market_data['features']['imbalance']
            spread = market_data['features']['bid_ask_spread']
            
            price_prediction = model_predictions['price_prediction']
            regime = model_predictions['regime']
            confidence = model_predictions['confidence']
            
            # Current mid price
            mid_price = (orderbook['asks'][0][0] + orderbook['bids'][0][0]) / 2
            
            # Check if we should trade
            if not self._should_open_position(
                imbalance, spread, price_prediction, confidence, regime):
                return None
            
            # Determine trade side based on imbalance and prediction
            side = 'buy' if imbalance > 0 and price_prediction > 0 else 'sell'
            
            # Calculate trade size
            size = self._calculate_trade_size(symbol, mid_price, confidence)
            if size == 0:
                return None
            
            # Create signal
            signal = Signal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                side=side,
                price=mid_price,
                size=size,
                type='limit',
                time_in_force='IOC',  # Immediate-or-cancel for HFT
                confidence=confidence,
                metadata={
                    'regime': regime.name,
                    'imbalance': imbalance,
                    'prediction': price_prediction,
                    'spread': spread,
                    'volatility_scale': self.volatility_scale
                }
            )
            
            # Validate and store signal
            if self._validate_signal(signal):
                self.add_signal_to_history(signal)
                self._last_signals[symbol] = signal
                return signal
            
            return None
            
        except Exception as e:
            logger.error("market_update_error",
                        error=str(e),
                        symbol=symbol)
            return None
    
    async def on_trade_update(self, trade_update: Dict[str, Any]) -> None:
        """Process trade execution update.
        
        Args:
            trade_update: Trade execution details
        """
        try:
            symbol = trade_update['symbol']
            size = trade_update['size']
            side = trade_update['side']
            price = trade_update['price']
            
            # Update position
            position_delta = size if side == 'buy' else -size
            old_position = self._positions.get(symbol, 0)
            self._positions[symbol] = old_position + position_delta
            
            # Update P&L tracking
            trade_pnl = trade_update.get('realized_pnl', 0)
            self._daily_pnl += trade_pnl
            self._session_peak_pnl = max(self._session_peak_pnl, self._daily_pnl)
            
            # Update metrics
            DAILY_METRICS.labels(
                symbol=symbol,
                metric='trade_count'
            ).inc()
            
            DAILY_METRICS.labels(
                symbol=symbol,
                metric='volume'
            ).inc(size * price)
            
            # Store trade
            self._recent_trades.append(trade_update)
            
            # Limit trade history
            if len(self._recent_trades) > 1000:
                self._recent_trades = self._recent_trades[-1000:]
            
            logger.info("trade_executed",
                       symbol=symbol,
                       side=side,
                       size=size,
                       price=price,
                       pnl=trade_pnl,
                       daily_pnl=self._daily_pnl)
            
        except Exception as e:
            logger.error("trade_update_error",
                        error=str(e),
                        update=trade_update)
    
    async def on_order_update(self, order_update: Dict[str, Any]) -> None:
        """Process order status update.
        
        Args:
            order_update: Order status details
        """
        try:
            if order_update['status'] == 'canceled':
                # Handle unfilled orders
                symbol = order_update['symbol']
                if symbol in self._last_signals:
                    del self._last_signals[symbol]
            
            logger.info("order_updated",
                       order_id=order_update['order_id'],
                       status=order_update['status'])
            
        except Exception as e:
            logger.error("order_update_error",
                        error=str(e),
                        update=order_update)
    
    async def on_position_update(self, position_update: Dict[str, Any]) -> None:
        """Process position update.
        
        Args:
            position_update: Current position details
        """
        try:
            symbol = position_update['symbol']
            self._positions[symbol] = position_update['size']
            
            logger.info("position_updated",
                       symbol=symbol,
                       size=position_update['size'],
                       entry_price=position_update.get('entry_price'))
            
        except Exception as e:
            logger.error("position_update_error",
                        error=str(e),
                        update=position_update)
    
    def update_parameters(self,
                       market_regime: MarketRegime,
                       volatility: float
                       ) -> None:
        """Update strategy parameters based on market conditions.
        
        Args:
            market_regime: Current market regime
            volatility: Current volatility level
        """
        try:
            # Track old regime for transition monitoring
            old_regime = getattr(self, '_current_regime', None)
            self._current_regime = market_regime
            
            if old_regime and old_regime != market_regime:
                REGIME_TRANSITIONS.labels(
                    symbol=self.symbol,
                    from_regime=old_regime.name,
                    to_regime=market_regime.name
                ).inc()
            
            # Update volatility scaling
            self.volatility_scale = max(0.5, min(1.0, 1.0 - volatility))
            VOLATILITY_SCALE.labels(symbol=self.symbol).set(self.volatility_scale)
            
            # Adjust spread threshold based on volatility
            self.min_spread = max(
                self.config['min_spread'],
                volatility * 0.1  # Higher spreads needed in volatile markets
            )
            
            # Adjust position sizing based on regime
            if market_regime == MarketRegime.VOLATILE:
                self.max_position_size = self.config['max_position_size'] * 0.5
                self.confidence_threshold = 0.8  # Higher confidence needed
                self.min_imbalance = 0.25  # Stronger signals required
            
            elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                self.max_position_size = self.config['max_position_size'] * 0.75
                self.confidence_threshold = 0.7
                self.min_imbalance = 0.2
            
            elif market_regime == MarketRegime.RANGING_LOW_VOL:
                self.max_position_size = self.config['max_position_size']
                self.confidence_threshold = 0.6  # Lower confidence acceptable
                self.min_imbalance = 0.15  # More sensitive to imbalance
            
            else:  # Default/Unknown regime
                self.max_position_size = self.config['max_position_size'] * 0.75
                self.confidence_threshold = 0.7
                self.min_imbalance = 0.2
            
            # Update metrics
            POSITION_SCALING.labels(
                symbol=self.symbol,
                regime=market_regime.name
            ).set(self.max_position_size / self.config['max_position_size'])
            
            CONFIDENCE_THRESHOLDS.labels(
                symbol=self.symbol,
                regime=market_regime.name
            ).set(self.confidence_threshold)
            
            # Update drawdown metrics
            current_drawdown = (self._session_peak_pnl - self._daily_pnl) / max(1.0, self._session_peak_pnl)
            DRAWDOWN_METRICS.labels(
                symbol=self.symbol,
                type='current'
            ).set(current_drawdown)
            
            DRAWDOWN_METRICS.labels(
                symbol=self.symbol,
                type='max_session'
            ).set(max(current_drawdown, getattr(self, '_max_drawdown', 0)))
            
            # Update daily metrics
            DAILY_METRICS.labels(
                symbol=self.symbol,
                metric='pnl'
            ).set(self._daily_pnl)
            
            logger.info("parameters_updated",
                       regime=market_regime.name,
                       volatility=volatility,
                       volatility_scale=self.volatility_scale,
                       min_spread=self.min_spread,
                       max_position=self.max_position_size,
                       confidence_threshold=self.confidence_threshold,
                       min_imbalance=self.min_imbalance)
            
        except Exception as e:
            logger.error("parameter_update_error",
                        error=str(e),
                        regime=market_regime.name)