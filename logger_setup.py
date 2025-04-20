"""Logging and metrics configuration.

Sets up structured logging and Prometheus metrics reporting
for monitoring trading activity and system health.
"""
import os
import sys
import logging
from typing import Dict, Any
from datetime import datetime
import structlog
from prometheus_client import Counter, Gauge, push_to_gateway

# Initialize metrics
TRADE_COUNT = Counter('trades_total', 'Number of trades executed', ['symbol', 'side'])
POSITION_SIZE = Gauge('position_size', 'Current position size', ['symbol'])
PNL = Gauge('pnl', 'Profit and loss', ['symbol', 'type'])
ERROR_COUNT = Counter('errors_total', 'Number of errors', ['type'])
MODEL_PREDICTIONS = Counter('model_predictions', 'Model prediction metrics', 
                          ['symbol', 'regime', 'direction'])

def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)

def push_metrics(job: str) -> None:
    """Push metrics to Prometheus gateway.
    
    Args:
        job: Job identifier
    """
    try:
        gateway = os.getenv('PROMETHEUS_PUSH_GATEWAY')
        if gateway:
            push_to_gateway(gateway, job=job, registry=None)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("metrics_push_failed", error=str(e))

def log_trade(logger: structlog.BoundLogger,
              side: str,
              symbol: str,
              size: float,
              price: float,
              **kwargs: Any) -> None:
    """Log trade execution with metrics.
    
    Args:
        logger: Logger instance
        side: Trade side (buy/sell)
        symbol: Trading pair symbol
        size: Trade size
        price: Trade price
        kwargs: Additional log fields
    """
    # Log trade
    logger.info("trade_executed",
                side=side,
                symbol=symbol,
                size=size,
                price=price,
                **kwargs)
    
    # Update metrics
    TRADE_COUNT.labels(symbol=symbol, side=side).inc()
    
    # Push metrics
    push_metrics('trades')

def log_position_update(logger: structlog.BoundLogger,
                       symbol: str,
                       size: float,
                       **kwargs: Any) -> None:
    """Log position update with metrics.
    
    Args:
        logger: Logger instance
        symbol: Trading pair symbol
        size: New position size
        kwargs: Additional log fields
    """
    # Log update
    logger.info("position_updated",
                symbol=symbol,
                size=size,
                **kwargs)
    
    # Update metrics
    POSITION_SIZE.labels(symbol=symbol).set(size)
    
    # Push metrics
    push_metrics('positions')

def log_pnl_update(logger: structlog.BoundLogger,
                   symbol: str,
                   realized_pnl: float,
                   unrealized_pnl: float,
                   **kwargs: Any) -> None:
    """Log P&L update with metrics.
    
    Args:
        logger: Logger instance
        symbol: Trading pair symbol
        realized_pnl: Realized P&L
        unrealized_pnl: Unrealized P&L
        kwargs: Additional log fields
    """
    # Log update
    logger.info("pnl_updated",
                symbol=symbol,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                **kwargs)
    
    # Update metrics
    PNL.labels(symbol=symbol, type='realized').set(realized_pnl)
    PNL.labels(symbol=symbol, type='unrealized').set(unrealized_pnl)
    
    # Push metrics
    push_metrics('pnl')

def log_model_prediction(logger: structlog.BoundLogger,
                        symbol: str,
                        regime: str,
                        prediction: float,
                        confidence: float,
                        **kwargs: Any) -> None:
    """Log model prediction with metrics.
    
    Args:
        logger: Logger instance
        symbol: Trading pair symbol
        regime: Market regime
        prediction: Price prediction
        confidence: Prediction confidence
        kwargs: Additional log fields
    """
    direction = 'up' if prediction > 0 else 'down'
    
    # Log prediction
    logger.info("model_prediction",
                symbol=symbol,
                regime=regime,
                prediction=prediction,
                confidence=confidence,
                direction=direction,
                **kwargs)
    
    # Update metrics
    MODEL_PREDICTIONS.labels(
        symbol=symbol,
        regime=regime,
        direction=direction
    ).inc()
    
    # Push metrics
    push_metrics('predictions')

def log_error(logger: structlog.BoundLogger,
              error_type: str,
              error: Exception,
              **kwargs: Any) -> None:
    """Log error with metrics.
    
    Args:
        logger: Logger instance
        error_type: Type of error
        error: Exception instance
        kwargs: Additional log fields
    """
    # Log error
    logger.error(f"{error_type}_error",
                 error=str(error),
                 error_type=error_type,
                 exc_info=True,  # Add traceback information
                 **kwargs)
    
    # Update metrics
    ERROR_COUNT.labels(type=error_type).inc()
    
    # Push metrics
    push_metrics('errors')