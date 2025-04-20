"""Prometheus metrics configuration.

Defines and registers metrics for monitoring bot performance,
latencies, and system health.
"""
from prometheus_client import Counter, Gauge, Histogram, Summary
import prometheus_client

# Initialize metrics registry
metrics_registry = prometheus_client.CollectorRegistry()

# Market data metrics
MARKET_DATA_UPDATES = Counter(
    'market_data_updates_total',
    'Number of market data updates received',
    ['symbol', 'type'],
    registry=metrics_registry
)

MARKET_DATA_LATENCY = Histogram(
    'market_data_latency_seconds',
    'Market data processing latency',
    ['symbol'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=metrics_registry
)

# Order execution metrics
ORDER_EVENTS = Counter(
    'order_events_total',
    'Number of order events',
    ['symbol', 'side', 'type', 'status'],
    registry=metrics_registry
)

EXECUTION_LATENCY = Histogram(
    'order_execution_latency_seconds',
    'Order execution latency',
    ['symbol', 'type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=metrics_registry
)

SLIPPAGE = Histogram(
    'order_slippage',
    'Order execution slippage',
    ['symbol', 'side'],
    buckets=(-0.01, -0.005, -0.001, 0.001, 0.005, 0.01),
    registry=metrics_registry
)

# Risk metrics
POSITION_VALUE = Gauge(
    'position_value',
    'Current position value',
    ['symbol'],
    registry=metrics_registry
)

PNL = Gauge(
    'pnl',
    'Profit and loss',
    ['symbol', 'type'],  # type: realized/unrealized
    registry=metrics_registry
)

DRAWDOWN = Gauge(
    'drawdown',
    'Current drawdown percentage',
    ['symbol'],
    registry=metrics_registry
)

RISK_EVENTS = Counter(
    'risk_events_total',
    'Number of risk management events',
    ['type', 'symbol'],
    registry=metrics_registry
)

# Strategy metrics
SIGNALS_GENERATED = Counter(
    'trading_signals_total',
    'Number of trading signals generated',
    ['symbol', 'side', 'strategy'],
    registry=metrics_registry
)

SIGNAL_CONFIDENCE = Histogram(
    'signal_confidence',
    'Trading signal confidence scores',
    ['symbol', 'strategy'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=metrics_registry
)

# Enhanced strategy metrics
REGIME_TRANSITIONS = Counter(
    'regime_transitions_total',
    'Number of market regime transitions',
    ['symbol', 'from_regime', 'to_regime'],
    registry=metrics_registry
)

VOLATILITY_SCALE = Gauge(
    'volatility_scale',
    'Current volatility scaling factor',
    ['symbol'],
    registry=metrics_registry
)

POSITION_SCALING = Gauge(
    'position_scaling',
    'Current position size scaling factor',
    ['symbol', 'regime'],
    registry=metrics_registry
)

CONFIDENCE_THRESHOLDS = Gauge(
    'confidence_thresholds',
    'Current confidence thresholds',
    ['symbol', 'regime'],
    registry=metrics_registry
)

DAILY_METRICS = Gauge(
    'daily_trading_metrics',
    'Daily trading performance metrics',
    ['symbol', 'metric'],  # metric: pnl, trades, volume, etc
    registry=metrics_registry
)

DRAWDOWN_METRICS = Gauge(
    'drawdown_metrics',
    'Drawdown related metrics',
    ['symbol', 'type'],  # type: current, max_session
    registry=metrics_registry
)

# ML model metrics
MODEL_PREDICTIONS = Counter(
    'model_predictions_total',
    'Number of model predictions',
    ['symbol', 'model', 'direction'],
    registry=metrics_registry
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency',
    ['symbol', 'model'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=metrics_registry
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model prediction accuracy',
    ['symbol', 'model', 'window'],
    registry=metrics_registry
)

# System metrics
ERROR_EVENTS = Counter(
    'error_events_total',
    'Number of error events',
    ['type', 'component'],
    registry=metrics_registry
)

SYSTEM_LATENCY = Summary(
    'system_latency_seconds',
    'Overall system processing latency',
    ['operation'],
    registry=metrics_registry
)