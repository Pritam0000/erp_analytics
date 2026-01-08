# src/models/predictive/base_predictor.py
"""Base class for all predictive models"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os


class BasePredictor(ABC):
    """Abstract base class for predictive models"""

    def __init__(self, model_name: str):
        """
        Initialize predictor

        Args:
            model_name: Name for this predictor
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Models
        self.rf_model = None
        self.xgb_model = None

        # Metadata
        self.feature_columns = []
        self.is_trained = False
        self.training_metrics = {}

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        pass

    @abstractmethod
    def train_random_forest(self, X_train, y_train) -> Dict:
        """Train Random Forest model"""
        pass

    @abstractmethod
    def train_xgboost(self, X_train, y_train) -> Dict:
        """Train XGBoost model"""
        pass

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_type: str = 'xgboost'
    ) -> Dict:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test targets
            model_type: 'rf' or 'xgboost'

        Returns:
            Dictionary of evaluation metrics
        """
        model = self.xgb_model if model_type == 'xgboost' else self.rf_model

        if model is None:
            raise ValueError(f"{model_type} model not trained")

        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)
        self.logger.info(f"{model_type.upper()} Metrics: {metrics}")

        return metrics

    @abstractmethod
    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        """Calculate model-specific metrics"""
        pass

    def predict(
        self,
        X: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features DataFrame
            model_type: 'rf' or 'xgboost'

        Returns:
            Array of predictions
        """
        model = self.xgb_model if model_type == 'xgboost' else self.rf_model

        if model is None:
            raise ValueError(f"{model_type} model not trained")

        return model.predict(X)

    def compare_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare RF and XGBoost performance

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame comparing model metrics
        """
        rf_metrics = self.evaluate(X_test, y_test, 'rf')
        xgb_metrics = self.evaluate(X_test, y_test, 'xgboost')

        comparison = pd.DataFrame({
            'Random Forest': rf_metrics,
            'XGBoost': xgb_metrics
        })

        return comparison

    def get_feature_importance(self, model_type: str = 'xgboost', top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            model_type: 'rf' or 'xgboost'
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        model = self.xgb_model if model_type == 'xgboost' else self.rf_model

        if model is None:
            raise ValueError(f"{model_type} model not trained")

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            raise AttributeError("Model doesn't have feature_importances_")

        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return feature_importance

    def save(self, model_dir: str):
        """
        Save trained models

        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)

        if self.rf_model:
            rf_path = os.path.join(model_dir, f'{self.model_name}_rf.pkl')
            joblib.dump(self.rf_model, rf_path)
            self.logger.info(f"Saved RF model to {rf_path}")

        if self.xgb_model:
            xgb_path = os.path.join(model_dir, f'{self.model_name}_xgb.pkl')
            joblib.dump(self.xgb_model, xgb_path)
            self.logger.info(f"Saved XGBoost model to {xgb_path}")

        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'training_metrics': self.training_metrics
        }
        metadata_path = os.path.join(model_dir, f'{self.model_name}_metadata.pkl')
        joblib.dump(metadata, metadata_path)

    def load(self, model_dir: str):
        """
        Load trained models

        Args:
            model_dir: Directory containing saved models
        """
        rf_path = os.path.join(model_dir, f'{self.model_name}_rf.pkl')
        xgb_path = os.path.join(model_dir, f'{self.model_name}_xgb.pkl')
        metadata_path = os.path.join(model_dir, f'{self.model_name}_metadata.pkl')

        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            self.logger.info(f"Loaded RF model from {rf_path}")

        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            self.logger.info(f"Loaded XGBoost model from {xgb_path}")

        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_columns = metadata['feature_columns']
            self.training_metrics = metadata.get('training_metrics', {})

        self.is_trained = (self.rf_model is not None or self.xgb_model is not None)
