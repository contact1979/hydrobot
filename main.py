"""Main entry point for HFT scalping bot.

Initializes and coordinates all trading components including data streams,
ML models, execution, risk management, and monitoring.
"""
import asyncio
import signal
from typing import Dict, Any, Set
import argparse
from datetime import datetime
from pathlib import Path
from prometheus_client import start_http_server

# Import settings and components using absolute paths
from hydrobot.config.settings import settings, load_secrets_into_settings
from hydrobot.data.market_data_stream import MarketDataStream
from hydrobot.data.client_binance import ExchangeClient
from hydrobot.trading.order_executor import OrderExecutor
from hydrobot.models.trainer import ModelTrainer
from hydrobot.models.inference import ModelInference
from hydrobot.trading.position_manager import PositionManager
from hydrobot.trading.risk_controller import RiskController
from hydrobot.strategies.strategy_manager import StrategyManager
from hydrobot.utils.logger_setup import setup_logging, get_logger

# Use the central logger
logger = get_logger(__name__)

class TradingBot:
    """Main trading bot coordinator."""
    
    def __init__(self):
        """Initialize trading bot using central settings."""
        # Initialize components using settings object
        self.exchange = ExchangeClient()
        self.market_data = MarketDataStream()
        self.position_manager = PositionManager()
        self.risk_controller = RiskController(
            risk_settings=settings.risk,
            position_manager=self.position_manager
        )
        self.order_executor = OrderExecutor(
            execution_settings=settings.execution,
            exchange=self.exchange
        )
        self.model_trainer = ModelTrainer(settings.ml)
        self.model_inference = ModelInference(settings.ml)
        self.strategy_manager = StrategyManager(settings.strategy)
        
        # Track active symbols
        self.active_symbols: Set[str] = set()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        # Start Prometheus metrics server if enabled
        metrics_port = getattr(settings.monitoring, 'metrics_port', None)
        if metrics_port:
            start_http_server(metrics_port)
            logger.info(f"Started metrics server on port {metrics_port}")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("Shutdown signal received", signal=signum)
        asyncio.create_task(self.shutdown())

    async def initialize(self) -> bool:
        """Initialize all components.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize exchange connection
            if not await self.exchange.initialize():
                return False
            
            # Load ML models
            if not await self.model_trainer.load_latest_models():
                return False
            
            # Initialize market data streams
            trading_pairs = settings.exchange.trading_pairs
            await self.market_data.initialize(trading_pairs)
            
            # Initialize order executor
            await self.order_executor.initialize()
            
            # Initialize position tracking
            for symbol in trading_pairs:
                self.position_manager.initialize_symbol(symbol)
            
            # Initialize strategies
            if not await self.strategy_manager.initialize():
                return False
            
            # Register callbacks
            self.market_data.register_callback(self._handle_market_update)
            
            logger.info("Bot initialized", 
                       symbols=trading_pairs, 
                       settings=settings.dict(exclude_secrets=True))
            
            return True
            
        except Exception as e:
            logger.error("Initialization error", error=str(e))
            return False

    async def _handle_market_update(self, market_data: Dict[str, Any]) -> None:
        """Process market data update.
        
        Args:
            market_data: Market data update
        """
        try:
            symbol = market_data['symbol']
            
            # Get model predictions
            predictions = await self.model_inference.get_predictions(
                market_data['features']
            )
            
            # Check if trading allowed
            if not self.risk_controller.can_trade(symbol):
                return
            
            # Generate trading signal
            signal = await self.strategy_manager.process_market_update(
                symbol,
                market_data,
                predictions
            )
            
            # Execute signal
            if signal:
                result = await self.order_executor.execute_signal(signal)
                
                if result.success:
                    # Update position
                    self.position_manager.update_position(
                        symbol,
                        {
                            'side': signal.side,
                            'size': signal.size,
                            'price': result.filled_price
                        },
                        market_data['orderbook']['mid_price']
                    )
                else:
                    logger.warning("Signal execution failed",
                                error=result.error,
                                signal=signal.__dict__)
            
        except Exception as e:
            logger.error("Market update error",
                        error=str(e),
                        data=market_data)
            self.risk_controller.record_error()

    async def start(self) -> None:
        """Start trading bot."""
        # Start market data streams
        await self.market_data.start()
        
        # Start strategies
        await self.strategy_manager.start_all()
        
        logger.info("Bot started")
    
    async def shutdown(self) -> None:
        """Gracefully shut down bot."""
        try:
            # Stop strategies
            await self.strategy_manager.stop_all()
            
            # Stop market data
            await self.market_data.stop()
            
            # Cancel all orders
            await self.order_executor.cancel_all_orders()
            
            # Close connections
            await self.order_executor.close()
            await self.exchange.close()
            
            logger.info("Bot shutdown complete")
            
        except Exception as e:
            logger.error("Shutdown error", error=str(e))

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HFT Scalping Bot')
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    args = parser.parse_args()
    
    # Setup logging using settings or CLI override
    log_level = args.log_level or settings.log_level
    setup_logging(log_level)
    
    # Load secrets into settings
    await load_secrets_into_settings()
    
    # Initialize and start bot
    bot = TradingBot()
    if await bot.initialize():
        await bot.start()
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Bot execution cancelled")
            await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())