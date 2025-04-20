"""Exchange client for market data and trading.

Provides a unified interface for exchange interactions using CCXT,
handling market data streaming and order execution.
"""
import asyncio
from typing import Dict, Any, List, Optional, Callable
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import json
import websockets
from utilities.logger_setup import get_logger

logger = get_logger(__name__)

class ExchangeClient:
    def __init__(self, config: Dict[str, Any]):
        """Initialize exchange client.
        
        Args:
            config: Exchange configuration
        """
        self.config = config
        self.exchange_id = config['name']
        self.testnet = config.get('testnet', False)
        
        # Initialize CCXT exchange
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Use testnet if configured
        if self.testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Websocket connections
        self._ws_connections: Dict[str, Any] = {}
        self._orderbook_callbacks: Dict[str, List[Callable]] = {}
        self._trade_callbacks: Dict[str, List[Callable]] = {}
        
        # Rate limiting
        self._last_request = datetime.utcnow()
        self._request_interval = timedelta(
            seconds=1 / config.get('requests_per_second', 10)
        )
    
    async def initialize(self) -> bool:
        """Initialize exchange connection.
        
        Returns:
            True if initialization successful
        """
        try:
            # Load markets
            await self.exchange.load_markets()
            
            # Test API connectivity
            await self.exchange.fetch_balance()
            
            logger.info("exchange_initialized",
                       exchange=self.exchange_id,
                       testnet=self.testnet)
            return True
            
        except Exception as e:
            logger.error("exchange_init_error",
                        error=str(e),
                        exchange=self.exchange_id)
            return False
    
    async def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        now = datetime.utcnow()
        elapsed = now - self._last_request
        if elapsed < self._request_interval:
            await asyncio.sleep(
                (self._request_interval - elapsed).total_seconds()
            )
        self._last_request = now
    
    async def get_orderbook(self,
                          symbol: str,
                          limit: int = 10
                          ) -> Dict[str, Any]:
        """Get current orderbook snapshot.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels
        
        Returns:
            Orderbook data
        """
        try:
            await self._respect_rate_limit()
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'bids': orderbook['bids'],
                'asks': orderbook['asks']
            }
            
        except Exception as e:
            logger.error("orderbook_fetch_error",
                        error=str(e),
                        symbol=symbol)
            return {}
    
    async def get_recent_trades(self,
                             symbol: str,
                             limit: int = 100
                             ) -> List[Dict[str, Any]]:
        """Get recent trades.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades
        
        Returns:
            List of recent trades
        """
        try:
            await self._respect_rate_limit()
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            
            return [{
                'symbol': trade['symbol'],
                'id': trade['id'],
                'timestamp': trade['timestamp'],
                'side': trade['side'],
                'price': trade['price'],
                'amount': trade['amount']
            } for trade in trades]
            
        except Exception as e:
            logger.error("trades_fetch_error",
                        error=str(e),
                        symbol=symbol)
            return []
    
    async def create_order(self,
                         symbol: str,
                         order_type: str,
                         side: str,
                         amount: float,
                         price: Optional[float] = None
                         ) -> Dict[str, Any]:
        """Create new order.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type (market/limit)
            side: Order side (buy/sell)
            amount: Order amount
            price: Optional limit price
        
        Returns:
            Order details
        """
        try:
            await self._respect_rate_limit()
            
            params = {}
            if order_type == 'limit':
                if not price:
                    raise ValueError("Price required for limit orders")
                params['price'] = price
            
            order = await self.exchange.create_order(
                symbol,
                order_type,
                side,
                amount,
                price,
                params
            )
            
            logger.info("order_created",
                       order_id=order['id'],
                       symbol=symbol,
                       type=order_type,
                       side=side,
                       amount=amount,
                       price=price)
            
            return order
            
        except Exception as e:
            logger.error("order_create_error",
                        error=str(e),
                        symbol=symbol,
                        type=order_type,
                        side=side)
            raise
    
    async def cancel_order(self,
                         order_id: str,
                         symbol: str
                         ) -> bool:
        """Cancel existing order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
        
        Returns:
            True if cancellation successful
        """
        try:
            await self._respect_rate_limit()
            await self.exchange.cancel_order(order_id, symbol)
            
            logger.info("order_canceled",
                       order_id=order_id,
                       symbol=symbol)
            return True
            
        except Exception as e:
            logger.error("order_cancel_error",
                        error=str(e),
                        order_id=order_id,
                        symbol=symbol)
            return False
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Position details
        """
        try:
            await self._respect_rate_limit()
            balance = await self.exchange.fetch_balance()
            
            # Extract position from balance
            currency = symbol.split('/')[0]
            free = balance[currency]['free']
            used = balance[currency]['used']
            total = balance[currency]['total']
            
            return {
                'symbol': symbol,
                'size': total,
                'free': free,
                'used': used
            }
            
        except Exception as e:
            logger.error("position_fetch_error",
                        error=str(e),
                        symbol=symbol)
            return {}
    
    async def subscribe_orderbook(self,
                               symbol: str,
                               callback: Callable[[Dict[str, Any]], None]
                               ) -> None:
        """Subscribe to orderbook updates.
        
        Args:
            symbol: Trading pair symbol
            callback: Update callback function
        """
        if symbol not in self._orderbook_callbacks:
            self._orderbook_callbacks[symbol] = []
        self._orderbook_callbacks[symbol].append(callback)
        
        # Start websocket if needed
        if symbol not in self._ws_connections:
            asyncio.create_task(
                self._maintain_websocket(symbol)
            )
    
    async def subscribe_trades(self,
                            symbol: str,
                            callback: Callable[[Dict[str, Any]], None]
                            ) -> None:
        """Subscribe to trade updates.
        
        Args:
            symbol: Trading pair symbol
            callback: Update callback function
        """
        if symbol not in self._trade_callbacks:
            self._trade_callbacks[symbol] = []
        self._trade_callbacks[symbol].append(callback)
        
        # Start websocket if needed
        if symbol not in self._ws_connections:
            asyncio.create_task(
                self._maintain_websocket(symbol)
            )
    
    async def _maintain_websocket(self, symbol: str) -> None:
        """Maintain websocket connection.
        
        Args:
            symbol: Trading pair symbol
        """
        while True:
            try:
                # Get websocket URL and subscribe message
                ws_url = self.exchange.urls['ws']
                subscribe_msg = {
                    'method': 'SUBSCRIBE',
                    'params': [
                        f"{symbol.lower()}@depth",
                        f"{symbol.lower()}@trade"
                    ],
                    'id': 1
                }
                
                async with websockets.connect(ws_url) as ws:
                    self._ws_connections[symbol] = ws
                    
                    # Subscribe
                    await ws.send(json.dumps(subscribe_msg))
                    
                    # Process messages
                    async for message in ws:
                        data = json.loads(message)
                        
                        if 'e' in data:  # Event type
                            if data['e'] == 'depthUpdate':
                                # Process orderbook update
                                update = {
                                    'symbol': symbol,
                                    'timestamp': data['E'],
                                    'bids': data['b'],
                                    'asks': data['a']
                                }
                                for callback in self._orderbook_callbacks.get(symbol, []):
                                    await callback(update)
                            
                            elif data['e'] == 'trade':
                                # Process trade update
                                update = {
                                    'symbol': symbol,
                                    'id': data['t'],
                                    'timestamp': data['E'],
                                    'price': float(data['p']),
                                    'amount': float(data['q']),
                                    'side': 'buy' if data['m'] else 'sell'
                                }
                                for callback in self._trade_callbacks.get(symbol, []):
                                    await callback(update)
                    
            except Exception as e:
                logger.error("websocket_error",
                           error=str(e),
                           symbol=symbol)
                
                # Remove connection
                self._ws_connections.pop(symbol, None)
                
                # Wait before reconnecting
                await asyncio.sleep(5)
    
    async def close(self) -> None:
        """Close exchange connection."""
        try:
            # Close websockets
            for ws in self._ws_connections.values():
                await ws.close()
            
            # Close exchange
            await self.exchange.close()
            
            logger.info("exchange_connection_closed",
                       exchange=self.exchange_id)
            
        except Exception as e:
            logger.error("exchange_close_error",
                        error=str(e),
                        exchange=self.exchange_id)