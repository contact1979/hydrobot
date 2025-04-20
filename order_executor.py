"""Order execution manager.

Handles order placement, cancellation, and execution monitoring
with non-blocking I/O and advanced execution algorithms.
"""
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
from strategies.base_strategy import Signal
from data_ingestion.exchange_client import ExchangeClient
from utilities.logger_setup import get_logger
from utilities.metrics import (
    ORDER_EVENTS,
    EXECUTION_LATENCY,
    SLIPPAGE,
    SYSTEM_LATENCY
)

logger = get_logger(__name__)

@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str]
    filled_price: Optional[float]
    filled_amount: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]

class OrderExecutor:
    def __init__(self, config: Dict[str, Any], exchange_client: ExchangeClient):
        """Initialize order executor.
        
        Args:
            config: Execution configuration
            exchange_client: Exchange client instance
        """
        self.config = config
        self.exchange = exchange_client
        
        # Execution settings
        self.max_slippage = config.get('max_slippage', 0.002)  # 0.2%
        self.order_timeout = config.get('order_timeout', 5)  # 5 seconds
        self.max_retries = config.get('max_retries', 3)
        
        # Async state
        self._order_updates: Dict[str, asyncio.Queue] = {}
        self._active_orders: Dict[str, Dict[str, Any]] = {}
        self._pending_cancels: Set[str] = set()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize executor resources."""
        self._session = aiohttp.ClientSession()
    
    def _calculate_limit_price(self, signal: Signal) -> float:
        """Calculate limit price with slippage control.
        
        Args:
            signal: Trading signal
        
        Returns:
            Limit price for order
        """
        # Add/subtract slippage from mid price
        slippage_factor = 1 + (self.max_slippage if signal.side == 'buy' else -self.max_slippage)
        return signal.price * slippage_factor
    
    @SYSTEM_LATENCY.labels(operation='wait_for_fill').time()
    async def _wait_for_fill(self, order_id: str, timeout: int) -> OrderResult:
        """Wait for order fill or timeout.
        
        Args:
            order_id: Exchange order ID
            timeout: Timeout in seconds
        
        Returns:
            Order execution result
        """
        try:
            # Create update queue if needed
            if order_id not in self._order_updates:
                self._order_updates[order_id] = asyncio.Queue()
            
            # Wait for updates with timeout
            start_time = datetime.utcnow()
            update = None
            
            while (datetime.utcnow() - start_time) < timedelta(seconds=timeout):
                try:
                    update = await asyncio.wait_for(
                        self._order_updates[order_id].get(),
                        timeout=0.1
                    )
                    
                    if update['status'] == 'filled':
                        # Record metrics
                        EXECUTION_LATENCY.labels(
                            symbol=update['symbol'],
                            type=update['type']
                        ).observe(
                            (datetime.utcnow() - start_time).total_seconds()
                        )
                        
                        slippage = (
                            update['price'] / self._active_orders[order_id]['price'] - 1
                            if self._active_orders[order_id]['side'] == 'buy'
                            else 1 - update['price'] / self._active_orders[order_id]['price']
                        )
                        SLIPPAGE.labels(
                            symbol=update['symbol'],
                            side=update['side']
                        ).observe(slippage)
                        
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            filled_price=update['price'],
                            filled_amount=update['filled'],
                            error=None,
                            metadata=update
                        )
                    
                    elif update['status'] == 'canceled':
                        return OrderResult(
                            success=False,
                            order_id=order_id,
                            filled_price=None,
                            filled_amount=update.get('filled', 0),
                            error='Order canceled',
                            metadata=update
                        )
                    
                except asyncio.TimeoutError:
                    continue
            
            # Timeout reached, cancel order
            await self.cancel_order(order_id)
            return OrderResult(
                success=False,
                order_id=order_id,
                filled_price=None,
                filled_amount=0,
                error='Order timeout',
                metadata={}
            )
            
        finally:
            # Cleanup
            if order_id in self._order_updates:
                del self._order_updates[order_id]
    
    @SYSTEM_LATENCY.labels(operation='execute_signal').time()
    async def execute_signal(self, signal: Signal) -> OrderResult:
        """Execute trading signal.
        
        Args:
            signal: Trading signal to execute
        
        Returns:
            Order execution result
        """
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                # Calculate limit price
                limit_price = self._calculate_limit_price(signal)
                
                # Place order with retry logic
                order = await self.exchange.create_order(
                    symbol=signal.symbol,
                    order_type=signal.type,
                    side=signal.side,
                    amount=signal.size,
                    price=limit_price if signal.type == 'limit' else None
                )
                
                order_id = order['id']
                self._active_orders[order_id] = order
                
                # Record metrics
                ORDER_EVENTS.labels(
                    symbol=signal.symbol,
                    side=signal.side,
                    type=signal.type,
                    status='placed'
                ).inc()
                
                logger.info("order_placed",
                          order_id=order_id,
                          symbol=signal.symbol,
                          side=signal.side,
                          size=signal.size,
                          price=limit_price,
                          type=signal.type)
                
                # Wait for fill
                return await self._wait_for_fill(order_id, self.order_timeout)
                
            except Exception as e:
                attempts += 1
                last_error = str(e)
                
                # Record error metric
                ORDER_EVENTS.labels(
                    symbol=signal.symbol,
                    side=signal.side,
                    type=signal.type,
                    status='error'
                ).inc()
                
                logger.error("order_execution_error",
                           error=str(e),
                           attempt=attempts,
                           signal=signal.__dict__)
                
                if attempts < self.max_retries:
                    await asyncio.sleep(0.5 * attempts)  # Exponential backoff
        
        return OrderResult(
            success=False,
            order_id=None,
            filled_price=None,
            filled_amount=0,
            error=f"Max retries exceeded: {last_error}",
            metadata={}
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order.
        
        Args:
            order_id: Exchange order ID
        
        Returns:
            True if cancellation successful
        """
        try:
            if order_id not in self._active_orders:
                return False
            
            # Prevent duplicate cancels
            if order_id in self._pending_cancels:
                return False
            self._pending_cancels.add(order_id)
            
            try:
                order = self._active_orders[order_id]
                await self.exchange.cancel_order(order_id, order['symbol'])
                
                # Record metric
                ORDER_EVENTS.labels(
                    symbol=order['symbol'],
                    side=order['side'],
                    type=order['type'],
                    status='canceled'
                ).inc()
                
                logger.info("order_canceled",
                          order_id=order_id,
                          symbol=order['symbol'])
                return True
                
            finally:
                self._pending_cancels.remove(order_id)
            
        except Exception as e:
            logger.error("order_cancel_error",
                        error=str(e),
                        order_id=order_id)
            return False
        
        finally:
            if order_id in self._active_orders:
                del self._active_orders[order_id]
    
    async def handle_order_update(self, update: Dict[str, Any]) -> None:
        """Process order status update.
        
        Args:
            update: Order update from exchange
        """
        try:
            order_id = update['order_id']
            
            # Store update in active orders
            if order_id in self._active_orders:
                self._active_orders[order_id].update(update)
            
            # Forward update to waiting coroutine
            if order_id in self._order_updates:
                await self._order_updates[order_id].put(update)
            
            # Record metrics
            ORDER_EVENTS.labels(
                symbol=update['symbol'],
                side=update.get('side', 'unknown'),
                type=update.get('type', 'unknown'),
                status=update['status']
            ).inc()
            
            # Cleanup filled/canceled orders
            if update['status'] in ['filled', 'canceled']:
                if order_id in self._active_orders:
                    del self._active_orders[order_id]
            
        except Exception as e:
            logger.error("order_update_error",
                        error=str(e),
                        update=update)
    
    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active orders.
        
        Returns:
            Dict of active orders
        """
        return self._active_orders.copy()
    
    async def cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        for order_id in list(self._active_orders.keys()):
            await self.cancel_order(order_id)
        
        logger.info("all_orders_canceled",
                   count=len(self._active_orders))
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.cancel_all_orders()
        
        if self._session:
            await self._session.close()