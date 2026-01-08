# src/analytics/predictive_models/__init__.py

from .cash_flow_forecaster import CashFlowForecaster
from .payment_predictor import PaymentPredictor
from .depreciation_forecaster import DepreciationForecaster
from .balance_predictor import BalancePredictor

# Version info
__version__ = '0.1.0'

# Model default parameters
DEFAULT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'confidence_level': 0.95
}

# Available model types
AVAILABLE_MODELS = {
    'cash_flow': CashFlowForecaster,
    'payment': PaymentPredictor,
    'depreciation': DepreciationForecaster,
    'balance': BalancePredictor
}

# Export classes and constants
__all__ = [
    'CashFlowForecaster',
    'PaymentPredictor',
    'DepreciationForecaster',
    'BalancePredictor',
    'DEFAULT_PARAMS',
    'AVAILABLE_MODELS'
]