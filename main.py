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
import yaml
from prometheus_client import start_http_server

from config.settings import settings
from data_ingestion.exchange_client import ExchangeClient
from data_ingestion.market_data_stream import MarketDataStream
from execution.order_executor import OrderExecutor
from ml_models.trainer import ModelTrainer
from ml_models.inference import ModelInference
from risk_management.position_manager import PositionManager
from risk_management.risk_controller import RiskController
from strategies.strategy_manager import StrategyManager
from utilities.logger_setup import setup_logging, get_logger

logger = get_logger(__name__)

class TradingBot:
    """Main trading bot coordinator."""
    
    def __init__(self, config_path: str):
        """Initialize trading bot.
        
        Args:
            config_path: Path to config file
        """
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Load environment-specific settings
        settings.load_secrets()
        
        # Initialize components
        self.exchange = ExchangeClient(self.config['exchange'])
        self.market_data = MarketDataStream(self.config['market_data'])
        self.position_manager = PositionManager(self.config['trading'])
        self.risk_controller = RiskController(
            self.config['risk'],
            self.position_manager
        )
        self.order_executor = OrderExecutor(
            self.config['execution'],
            self.exchange
        )
        self.model_trainer = ModelTrainer(self.config['ml'])
        self.model_inference = ModelInference(self.config['ml'])
        self.strategy_manager = StrategyManager(self.config['strategy'])
        
        # Track active symbols
        self.active_symbols: Set[str] = set()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        # Start Prometheus metrics server
        start_http_server(
            self.config['monitoring']['metrics_port']
        )
    
    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("shutdown_signal_received", signal=signum)
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
            symbols = self.config['trading']['trading_pairs']
            await self.market_data.initialize(symbols)
            
            # Initialize order executor
            await self.order_executor.initialize()
            
            # Initialize position tracking
            for symbol in symbols:
                self.position_manager.initialize_symbol(symbol)
            
            # Initialize strategies
            if not await self.strategy_manager.initialize():
                return False
            
            # Register callbacks
            self.market_data.register_callback(self._handle_market_update)
            
            logger.info("bot_initialized",
                       symbols=symbols,
                       config=self.config)
            
            return True
            
        except Exception as e:
            logger.error("initialization_error", error=str(e))
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
                    logger.warning("signal_execution_failed",
                                error=result.error,
                                signal=signal.__dict__)
            
        except Exception as e:
            logger.error("market_update_error",
                        error=str(e),
                        data=market_data)
            self.risk_controller.record_error()
    
    async def start(self) -> None:
        """Start trading bot."""
        # Start market data streams
        await self.market_data.start()
        
        # Start strategies
        await self.strategy_manager.start_all()
        
        logger.info("bot_started")
    
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
            
            logger.info("bot_shutdown_complete")
            
        except Exception as e:
            logger.error("shutdown_error", error=str(e))

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HFT Scalping Bot')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Initialize and start bot
    bot = TradingBot(args.config)
    if await bot.initialize():
        await bot.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())