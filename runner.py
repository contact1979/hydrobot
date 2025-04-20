"""Backtesting engine runner.

Executes trading strategies on historical data and produces
performance metrics for strategy evaluation.
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from .data_loader import HistoricalDataLoader
from strategies.scalping_strategy import ScalpingStrategy
from ml_models.inference import ModelInference
from utilities.logger_setup import get_logger

logger = get_logger(__name__)

class BacktestMetrics:
    """Backtesting performance metrics calculator."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, float] = {}
        self.cash = 0.0
        self.equity_curve: List[float] = []
        self.drawdowns: List[float] = []
        self.latencies: List[float] = []
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Record executed trade.
        
        Args:
            trade: Trade execution details
        """
        self.trades.append(trade)
        
        # Update position
        symbol = trade['symbol']
        size = trade['size'] * (1 if trade['side'] == 'buy' else -1)
        self.positions[symbol] = self.positions.get(symbol, 0) + size
        
        # Update cash
        self.cash -= size * trade['price']
        
        # Calculate equity
        equity = self.cash + sum(
            size * trade['price']
            for symbol, size in self.positions.items()
        )
        self.equity_curve.append(equity)
        
        # Calculate drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            dd = (peak - equity) / peak
            self.drawdowns.append(dd)
    
    def add_latency(self, latency: float) -> None:
        """Record strategy latency.
        
        Args:
            latency: Strategy execution latency in milliseconds
        """
        self.latencies.append(latency)
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary metrics.
        
        Returns:
            Dict of performance metrics
        """
        if not self.trades:
            return {}
        
        total_pnl = self.equity_curve[-1] - self.equity_curve[0]
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        
        return {
            'total_pnl': total_pnl,
            'return_pct': (total_pnl / self.equity_curve[0]) * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'trade_count': len(self.trades),
            'avg_trade_pnl': total_pnl / len(self.trades),
            'win_rate': sum(1 for t in self.trades if t['pnl'] > 0) / len(self.trades),
            'avg_latency_ms': np.mean(self.latencies),
            'p99_latency_ms': np.percentile(self.latencies, 99)
        }

class BacktestRunner:
    """Backtesting engine runner."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.data_loader = HistoricalDataLoader(config)
        self.metrics = BacktestMetrics()
        
        # Initialize components
        self.strategy = ScalpingStrategy(config)
        self.model_inference = ModelInference(config)
    
    async def _process_market_data(self,
                                market_data: Dict[str, Any],
                                timestamp: datetime
                                ) -> None:
        """Process market data batch.
        
        Args:
            market_data: Market data batch
            timestamp: Current timestamp
        """
        for symbol, data in market_data.items():
            # Track latency
            start_time = datetime.utcnow()
            
            # Get model predictions
            predictions = await self.model_inference.get_predictions(
                data['features'],
                {'prices': data['prices'], 'volumes': data['volumes']}
            )
            
            # Generate trading signal
            signal = await self.strategy.on_market_update(
                symbol,
                data,
                {'predictions': predictions}
            )
            
            # Record latency
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.add_latency(latency)
            
            # Execute signal
            if signal:
                # Simulate execution with basic slippage
                slippage = self.config.get('sim_slippage', 0.0001)
                executed_price = signal.price * (
                    1 + slippage if signal.side == 'buy' else 1 - slippage
                )
                
                trade = {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'side': signal.side,
                    'size': signal.size,
                    'price': executed_price,
                    'slippage': slippage,
                    'latency': latency
                }
                
                self.metrics.add_trade(trade)
    
    async def run(self,
                 symbols: List[str],
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None
                 ) -> Dict[str, Any]:
        """Run backtest.
        
        Args:
            symbols: Trading pair symbols
            start_time: Optional start time
            end_time: Optional end time
        
        Returns:
            Dict containing backtest results
        """
        try:
            logger.info("backtest_started",
                       symbols=symbols,
                       start=start_time,
                       end=end_time)
            
            # Initialize strategy
            await self.strategy.initialize()
            
            # Process historical data
            for batch in self.data_loader.iterate_market_data(symbols):
                timestamp = datetime.fromisoformat(
                    batch[symbols[0]]['timestamp']
                )
                
                # Skip if outside time range
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    break
                
                await self._process_market_data(batch, timestamp)
            
            # Get performance metrics
            results = self.metrics.get_summary()
            
            logger.info("backtest_completed",
                       metrics=results)
            
            return {
                'metrics': results,
                'trades': self.metrics.trades,
                'equity_curve': self.metrics.equity_curve
            }
            
        except Exception as e:
            logger.error("backtest_error",
                        error=str(e))
            return {}