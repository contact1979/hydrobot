"""Data processing and normalization module.

Handles cleaning and normalization of raw market data streams
for feature engineering and model input preparation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import deque
from utilities.logger_setup import get_logger

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, window_size: int = 100):
        """Initialize data processor with window settings.
        
        Args:
            window_size: Number of data points to maintain in rolling windows
        """
        self.window_size = window_size
        
        # Rolling windows for different data types
        self._orderbook_bids: Dict[str, deque] = {}
        self._orderbook_asks: Dict[str, deque] = {}
        self._trades: Dict[str, deque] = {}
        self._ticker_data: Dict[str, deque] = {}
        
        # Cache for processed features
        self._last_processed: Dict[str, Dict[str, float]] = {}
    
    def initialize_symbol(self, symbol: str) -> None:
        """Initialize data structures for a new symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        self._orderbook_bids[symbol] = deque(maxlen=self.window_size)
        self._orderbook_asks[symbol] = deque(maxlen=self.window_size)
        self._trades[symbol] = deque(maxlen=self.window_size)
        self._ticker_data[symbol] = deque(maxlen=self.window_size)
        self._last_processed[symbol] = {}
    
    def process_orderbook(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw orderbook data.
        
        Args:
            symbol: Trading pair symbol
            data: Raw orderbook update
        
        Returns:
            Processed orderbook features
        """
        try:
            # Extract bid/ask levels
            bids = np.array(data['bids'], dtype=float)
            asks = np.array(data['asks'], dtype=float)
            
            # Store in rolling windows
            self._orderbook_bids[symbol].append(bids)
            self._orderbook_asks[symbol].append(asks)
            
            # Calculate basic orderbook features
            bid_ask_spread = asks[0][0] - bids[0][0]
            bid_depth = np.sum(bids[:, 1])
            ask_depth = np.sum(asks[:, 1])
            
            # Calculate order book imbalance
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            processed = {
                'bid_ask_spread': bid_ask_spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance,
                'mid_price': (asks[0][0] + bids[0][0]) / 2
            }
            
            self._last_processed[symbol].update(processed)
            return processed
            
        except Exception as e:
            logger.error("orderbook_processing_error",
                        error=str(e),
                        symbol=symbol)
            return self._last_processed[symbol]
    
    def process_trades(self, symbol: str, trades: List[Dict]) -> Dict[str, float]:
        """Process raw trade data.
        
        Args:
            symbol: Trading pair symbol
            trades: List of recent trades
        
        Returns:
            Processed trade features
        """
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(trades)
            df['price'] = pd.to_numeric(df['price'])
            df['amount'] = pd.to_numeric(df['amount'])
            
            # Store in rolling window
            self._trades[symbol].append(df)
            
            # Calculate trade features
            vwap = (df['price'] * df['amount']).sum() / df['amount'].sum()
            buy_volume = df[df['side'] == 'buy']['amount'].sum()
            sell_volume = df[df['side'] == 'sell']['amount'].sum()
            
            # Calculate buy/sell pressure
            volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            
            processed = {
                'vwap': vwap,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'volume_imbalance': volume_imbalance
            }
            
            self._last_processed[symbol].update(processed)
            return processed
            
        except Exception as e:
            logger.error("trades_processing_error",
                        error=str(e),
                        symbol=symbol)
            return self._last_processed[symbol]
    
    def process_ticker(self, symbol: str, ticker: Dict[str, Any]) -> Dict[str, float]:
        """Process ticker data.
        
        Args:
            symbol: Trading pair symbol
            ticker: Ticker data
        
        Returns:
            Processed ticker features
        """
        try:
            # Store in rolling window
            self._ticker_data[symbol].append(ticker)
            
            # Calculate ticker features
            price_change = float(ticker['price_change'])
            volume = float(ticker['volume'])
            
            processed = {
                'price_change': price_change,
                'volume': volume,
                'price_change_percent': price_change / float(ticker['open_price'])
            }
            
            self._last_processed[symbol].update(processed)
            return processed
            
        except Exception as e:
            logger.error("ticker_processing_error",
                        error=str(e),
                        symbol=symbol)
            return self._last_processed[symbol]
    
    def get_feature_snapshot(self, symbol: str) -> Dict[str, float]:
        """Get latest processed features for symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Dict of latest processed features
        """
        return self._last_processed.get(symbol, {})