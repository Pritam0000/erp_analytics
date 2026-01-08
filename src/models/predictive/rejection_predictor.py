# src/models/predictive/rejection_predictor.py
"""Predict invoice rejection probability using RF and XGBoost"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

from .base_predictor import BasePredictor
from config.model_config import RF_CLASSIFIER_CONFIG, XGB_CLASSIFIER_CONFIG


class RejectionPredictor(BasePredictor):
    """Predict probability of invoice rejection"""

    def __init__(self):
        super().__init__(model_name='rejection_predictor')

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for rejection prediction

        Args:
            df: DataFrame with engineered features (must include is_rejected)

        Returns:
            Tuple of (X, y)
        """
        if 'is_rejected' not in df.columns:
            raise ValueError("DataFrame must contain 'is_rejected' column")

        # Select relevant features
        feature_cols = [
            'invoice_amount',
            'po_mismatch_flag',
            'po_invoice_variance_abs',
            'supplier_rejection_rate',
            'supplier_avg_amount',
            'is_high_value',
            'has_po',
            'is_rush',
            'tax_percentage',
            'amount_per_line_item',
            'supplier_avg_delay',
            'days_until_due',
            'day_of_week',
            'month',
            'invoice_category_encoded'
        ]

        # Filter columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols

        X = df[feature_cols].copy()
        X = X.fillna(0)

        # Target is is_rejected
        y = df['is_rejected'].copy()

        rejection_rate = y.mean()
        self.logger.info(f"Prepared {len(X)} samples. Rejection rate: {rejection_rate:.2%}")

        return X, y

    def train_random_forest(self, X_train, y_train) -> Dict:
        """Train Random Forest classifier"""
        self.logger.info("Training Random Forest classifier for rejection prediction...")

        self.rf_model = RandomForestClassifier(**RF_CLASSIFIER_CONFIG)
        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_train)
        y_prob = self.rf_model.predict_proba(X_train)[:, 1]
        metrics = self._calculate_metrics(y_train, y_pred, y_prob)

        self.logger.info(f"RF Training complete. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        return metrics

    def train_xgboost(self, X_train, y_train) -> Dict:
        """Train XGBoost classifier"""
        self.logger.info("Training XGBoost classifier for rejection prediction...")

        self.xgb_model = XGBClassifier(**XGB_CLASSIFIER_CONFIG)
        self.xgb_model.fit(X_train, y_train)

        y_pred = self.xgb_model.predict(X_train)
        y_prob = self.xgb_model.predict_proba(X_train)[:, 1]
        metrics = self._calculate_metrics(y_train, y_pred, y_prob)

        self.logger.info(f"XGBoost Training complete. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        self.is_trained = True
        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_prob=None) -> Dict:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def predict_proba(
        self,
        X: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> np.ndarray:
        """
        Predict rejection probabilities

        Args:
            X: Features DataFrame
            model_type: 'rf' or 'xgboost'

        Returns:
            Array of rejection probabilities (0 to 1)
        """
        model = self.xgb_model if model_type == 'xgboost' else self.rf_model

        if model is None:
            raise ValueError(f"{model_type} model not trained")

        # Return probability of class 1 (rejected)
        return model.predict_proba(X)[:, 1]

    def predict_with_risk_level(
        self,
        X: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> pd.DataFrame:
        """
        Predict rejection with risk levels

        Args:
            X: Features DataFrame
            model_type: 'rf' or 'xgboost'

        Returns:
            DataFrame with predictions, probabilities, and risk levels
        """
        predictions = self.predict(X, model_type)
        probabilities = self.predict_proba(X, model_type)

        # Categorize risk
        risk_levels = pd.cut(
            probabilities,
            bins=[-0.01, 0.3, 0.6, 1.01],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )

        result = pd.DataFrame({
            'will_be_rejected': predictions,
            'rejection_probability': probabilities,
            'risk_level': risk_levels,
            'recommendation': risk_levels.map({
                'Low Risk': 'Approve',
                'Medium Risk': 'Review',
                'High Risk': 'Investigate'
            })
        })

        return result
