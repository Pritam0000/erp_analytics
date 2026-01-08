# src/models/predictive/delay_predictor.py
"""Predict payment delays using RF and XGBoost"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from .base_predictor import BasePredictor
from config.model_config import RF_REGRESSOR_CONFIG, XGB_REGRESSOR_CONFIG


class DelayPredictor(BasePredictor):
    """Predict payment delay days"""

    def __init__(self):
        super().__init__(model_name='delay_predictor')

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for delay prediction

        Args:
            df: DataFrame with engineered features (must include payment_delay_days)

        Returns:
            Tuple of (X, y)
        """
        # Filter to only paid invoices (have actual payment delay)
        df_paid = df[df['payment_delay_days'].notna()].copy()

        if len(df_paid) == 0:
            raise ValueError("No paid invoices with payment_delay_days found")

        # Select relevant features
        feature_cols = [
            'invoice_amount',
            'payment_terms_days',
            'supplier_avg_delay',
            'supplier_rejection_rate',
            'supplier_avg_amount',
            'days_until_due',
            'has_po',
            'po_mismatch_flag',
            'is_high_value',
            'is_rush',
            'day_of_week',
            'month',
            'quarter',
            'invoice_category_encoded'
        ]

        # Filter columns that exist
        feature_cols = [col for col in feature_cols if col in df_paid.columns]
        self.feature_columns = feature_cols

        X = df_paid[feature_cols].copy()
        X = X.fillna(0)

        # Target is payment_delay_days
        y = df_paid['payment_delay_days'].copy()

        self.logger.info(f"Prepared {len(X)} paid invoices with {len(feature_cols)} features")

        return X, y

    def train_random_forest(self, X_train, y_train) -> Dict:
        """Train Random Forest regressor"""
        self.logger.info("Training Random Forest model for delay prediction...")

        self.rf_model = RandomForestRegressor(**RF_REGRESSOR_CONFIG)
        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)

        self.logger.info(f"RF Training complete. R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f} days")

        return metrics

    def train_xgboost(self, X_train, y_train) -> Dict:
        """Train XGBoost regressor"""
        self.logger.info("Training XGBoost model for delay prediction...")

        self.xgb_model = XGBRegressor(**XGB_REGRESSOR_CONFIG)
        self.xgb_model.fit(X_train, y_train)

        y_pred = self.xgb_model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)

        self.logger.info(f"XGBoost Training complete. R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f} days")

        self.is_trained = True
        return metrics

    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        """Calculate regression metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'median_error': np.median(np.abs(y_true - y_pred))
        }
        return metrics

    def predict_payment_date(
        self,
        X: pd.DataFrame,
        due_dates: pd.Series,
        model_type: str = 'xgboost'
    ) -> pd.DataFrame:
        """
        Predict expected payment dates

        Args:
            X: Features DataFrame
            due_dates: Series of due dates
            model_type: 'rf' or 'xgboost'

        Returns:
            DataFrame with predicted delay days and expected payment dates
        """
        delay_predictions = self.predict(X, model_type)

        result = pd.DataFrame({
            'predicted_delay_days': delay_predictions,
            'due_date': due_dates,
            'predicted_payment_date': due_dates + pd.to_timedelta(delay_predictions, unit='D'),
            'risk_level': pd.cut(
                delay_predictions,
                bins=[-np.inf, 0, 7, 30, np.inf],
                labels=['Early', 'On-Time', 'Moderate Delay', 'Severe Delay']
            )
        })

        return result
