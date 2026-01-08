# src/models/predictive/amount_predictor.py
"""Predict invoice amounts using RF and XGBoost"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

from .base_predictor import BasePredictor
from config.model_config import RF_REGRESSOR_CONFIG, XGB_REGRESSOR_CONFIG


class AmountPredictor(BasePredictor):
    """Predict invoice amounts based on supplier history and PO"""

    def __init__(self):
        super().__init__(model_name='amount_predictor')

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for amount prediction

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (X, y)
        """
        # Select relevant features for amount prediction
        feature_cols = [
            'supplier_avg_amount',
            'po_amount',
            'payment_terms_days',
            'line_items_count',
            'days_until_due',
            'tax_percentage',
            'supplier_rejection_rate',
            'supplier_invoice_count',
            'has_po',
            'is_rush',
            'day_of_week',
            'month',
            'quarter',
            'invoice_category_encoded'
        ]

        # Filter columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols

        X = df[feature_cols].copy()

        # Fill NaN values
        X = X.fillna(0)

        # Target is invoice_amount
        y = df['invoice_amount'].copy()

        self.logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")

        return X, y

    def train_random_forest(self, X_train, y_train) -> Dict:
        """
        Train Random Forest regressor

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Training Random Forest model...")

        self.rf_model = RandomForestRegressor(**RF_REGRESSOR_CONFIG)
        self.rf_model.fit(X_train, y_train)

        # Training predictions
        y_pred = self.rf_model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)

        self.logger.info(f"RF Training complete. R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")

        return metrics

    def train_xgboost(self, X_train, y_train) -> Dict:
        """
        Train XGBoost regressor

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Training XGBoost model...")

        self.xgb_model = XGBRegressor(**XGB_REGRESSOR_CONFIG)
        self.xgb_model.fit(X_train, y_train)

        # Training predictions
        y_pred = self.xgb_model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)

        self.logger.info(f"XGBoost Training complete. R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")

        self.is_trained = True
        return metrics

    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        """Calculate regression metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        return metrics

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        model_type: str = 'xgboost',
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Predict with confidence intervals

        Args:
            X: Features DataFrame
            model_type: 'rf' or 'xgboost'
            confidence_level: Confidence level for intervals

        Returns:
            DataFrame with predictions and confidence intervals
        """
        predictions = self.predict(X, model_type)

        # For Random Forest, use estimator predictions for confidence
        if model_type == 'rf' and self.rf_model:
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X) for tree in self.rf_model.estimators_])

            # Calculate std and confidence intervals
            std = np.std(tree_predictions, axis=0)
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%

            lower = predictions - z_score * std
            upper = predictions + z_score * std

        else:
            # For XGBoost, use simple percentage-based interval
            margin = predictions * 0.1  # 10% margin
            lower = predictions - margin
            upper = predictions + margin

        result = pd.DataFrame({
            'predicted_amount': predictions,
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_level': confidence_level
        })

        return result
