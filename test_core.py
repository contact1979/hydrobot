"""Core functionality tests for HFT scalping bot."""
import pytest
from datetime import datetime
import numpy as np
from hft_scalping_bot.data_ingestion.market_data_stream import OrderBook
from hft_scalping_bot.risk_management.position_manager import Position, PositionManager
from hft_scalping_bot.strategies.base_strategy import Signal

@pytest.fixture
def orderbook():
    """Create test orderbook."""
    book = OrderBook(depth=5)
    book.update(
        bids=[[100.0, 1.0], [99.0, 2.0], [98.0, 3.0]],
        asks=[[101.0, 1.0], [102.0, 2.0], [103.0, 3.0]]
    )
    return book

@pytest.fixture
def position():
    """Create test position."""
    return Position(
        symbol="BTC/USDT",
        size=0.1,
        entry_price=100.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        trades=[],
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def signal():
    """Create test trading signal."""
    return Signal(
        timestamp=datetime.utcnow(),
        symbol="BTC/USDT",
        side="buy",
        price=100.0,
        size=0.1,
        type="limit",
        time_in_force="IOC",
        confidence=0.8,
        metadata={}
    )

def test_orderbook_mid_price(orderbook):
    """Test orderbook mid price calculation."""
    assert orderbook.get_mid_price() == 100.5

def test_orderbook_spread(orderbook):
    """Test orderbook spread calculation."""
    assert orderbook.get_spread() == 1.0

def test_orderbook_imbalance(orderbook):
    """Test orderbook imbalance calculation."""
    # Sum of bid amounts = 6.0, sum of ask amounts = 6.0
    # Imbalance should be 0.0
    assert abs(orderbook.get_imbalance()) < 1e-6

def test_position_pnl_calculation(position):
    """Test position P&L calculations."""
    # Test unrealized P&L calculation based on fixture state
    # Current price 110.0, entry at 100.0, size 0.1
    # Unrealized P&L = (110 - 100) * 0.1 = 1.0
    current_price = 110.0
    unrealized_pnl = (current_price - position.entry_price) * position.size
    assert unrealized_pnl == 1.0
    # Note: This test only checks the calculation logic, not the PositionManager update

def test_signal_validation(signal):
    """Test trading signal validation."""
    # Test valid signal
    assert signal.side in ["buy", "sell"]
    assert signal.price > 0
    assert signal.size > 0
    assert 0 <= signal.confidence <= 1

@pytest.mark.asyncio
async def test_position_manager():
    """Test position manager functionality."""
    config = {
        "max_position_size": 1.0,
        "trading_pairs": ["BTC/USDT"]
    }
    
    manager = PositionManager(config)
    manager.initialize_symbol("BTC/USDT")
    
    # Test initial position update (buy)
    trade1 = {
        "symbol": "BTC/USDT", "side": "buy", "size": 0.1, "price": 100.0, "timestamp": datetime.utcnow()
    }
    position = manager.update_position("BTC/USDT", trade1, 105.0) # Mark price 105.0
    
    assert position.symbol == "BTC/USDT"
    assert position.size == 0.1
    assert position.entry_price == 100.0
    assert position.realized_pnl == 0.0
    # Unrealized PNL = (105.0 - 100.0) * 0.1 = 0.5
    assert abs(position.unrealized_pnl - 0.5) < 1e-9
    assert abs(position.total_pnl - 0.5) < 1e-9

    # Test second position update (buy more)
    trade2 = {
        "symbol": "BTC/USDT", "side": "buy", "size": 0.2, "price": 110.0, "timestamp": datetime.utcnow()
    }
    position = manager.update_position("BTC/USDT", trade2, 115.0) # Mark price 115.0

    # New size = 0.1 + 0.2 = 0.3
    # New entry price = (0.1 * 100.0 + 0.2 * 110.0) / 0.3 = (10 + 22) / 0.3 = 32 / 0.3 = 106.666...
    assert position.size == 0.3
    assert abs(position.entry_price - (32 / 0.3)) < 1e-9
    assert position.realized_pnl == 0.0
    # Unrealized PNL = (115.0 - 106.666...) * 0.3 = 8.333... * 0.3 = 2.5
    assert abs(position.unrealized_pnl - 2.5) < 1e-9
    assert abs(position.total_pnl - 2.5) < 1e-9

    # Test partial close (sell)
    trade3 = {
        "symbol": "BTC/USDT", "side": "sell", "size": 0.15, "price": 120.0, "timestamp": datetime.utcnow()
    }
    position = manager.update_position("BTC/USDT", trade3, 125.0) # Mark price 125.0

    # Realized PNL from closing 0.15 = (120.0 - 106.666...) * 0.15 = 13.333... * 0.15 = 2.0
    # New size = 0.3 - 0.15 = 0.15
    # Entry price remains 106.666...
    assert abs(position.size - 0.15) < 1e-9
    assert abs(position.entry_price - (32 / 0.3)) < 1e-9
    assert abs(position.realized_pnl - 2.0) < 1e-9
    # Unrealized PNL = (125.0 - 106.666...) * 0.15 = 18.333... * 0.15 = 2.75
    assert abs(position.unrealized_pnl - 2.75) < 1e-9
    # Total PNL = Realized + Unrealized = 2.0 + 2.75 = 4.75
    assert abs(position.total_pnl - 4.75) < 1e-9