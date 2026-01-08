# src/services/prediction_service.py
"""Service layer for predictive models"""

import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.data.data_loader import InvoiceDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.predictive.amount_predictor import AmountPredictor
from src.models.predictive.delay_predictor import DelayPredictor
from src.models.predictive.rejection_predictor import RejectionPredictor
from sklearn.model_selection import train_test_split


class PredictionService:
    """Orchestrate all predictive analytics"""

    def __init__(self, data_loader: Optional[InvoiceDataLoader] = None):
        """Initialize prediction service"""
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader or InvoiceDataLoader(use_mock_db=True)

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.amount_predictor = AmountPredictor()
        self.delay_predictor = DelayPredictor()
        self.rejection_predictor = RejectionPredictor()

        self.is_trained = False

    def train_all_models(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Train all predictive models

        Args:
            start_date: Training data start date
            end_date: Training data end date

        Returns:
            Dictionary with training results
        """
        self.logger.info("Training all predictive models...")

        # Load data
        invoices_df = self.data_loader.load_invoices(start_date=start_date, end_date=end_date)
        suppliers_df = self.data_loader.load_suppliers()

        # Feature engineering
        self.logger.info("Engineering features...")
        self.feature_engineer.fit(invoices_df, suppliers_df)
        featured_df = self.feature_engineer.transform(invoices_df, suppliers_df)

        results = {}

        # Train amount predictor
        try:
            X, y = self.amount_predictor.prepare_features(featured_df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.amount_predictor.train_random_forest(X_train, y_train)
            self.amount_predictor.train_xgboost(X_train, y_train)

            results['amount_predictor'] = {
                'rf_metrics': self.amount_predictor.evaluate(X_test, y_test, 'rf'),
                'xgb_metrics': self.amount_predictor.evaluate(X_test, y_test, 'xgboost')
            }
        except Exception as e:
            self.logger.error(f"Error training amount predictor: {str(e)}")
            results['amount_predictor'] = {'error': str(e)}

        # Train delay predictor
        try:
            X, y = self.delay_predictor.prepare_features(featured_df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.delay_predictor.train_random_forest(X_train, y_train)
            self.delay_predictor.train_xgboost(X_train, y_train)

            results['delay_predictor'] = {
                'rf_metrics': self.delay_predictor.evaluate(X_test, y_test, 'rf'),
                'xgb_metrics': self.delay_predictor.evaluate(X_test, y_test, 'xgboost')
            }
        except Exception as e:
            self.logger.error(f"Error training delay predictor: {str(e)}")
            results['delay_predictor'] = {'error': str(e)}

        # Train rejection predictor
        try:
            X, y = self.rejection_predictor.prepare_features(featured_df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.rejection_predictor.train_random_forest(X_train, y_train)
            self.rejection_predictor.train_xgboost(X_train, y_train)

            results['rejection_predictor'] = {
                'rf_metrics': self.rejection_predictor.evaluate(X_test, y_test, 'rf'),
                'xgb_metrics': self.rejection_predictor.evaluate(X_test, y_test, 'xgboost')
            }
        except Exception as e:
            self.logger.error(f"Error training rejection predictor: {str(e)}")
            results['rejection_predictor'] = {'error': str(e)}

        self.is_trained = True
        self.logger.info("All models trained successfully")

        return results

    def predict_invoice_amount(
        self,
        invoice_data: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> pd.DataFrame:
        """Predict invoice amounts"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")

        # Load suppliers for feature engineering
        suppliers_df = self.data_loader.load_suppliers()

        # Engineer features
        featured_df = self.feature_engineer.transform(invoice_data, suppliers_df)

        # Prepare and predict
        X, _ = self.amount_predictor.prepare_features(featured_df)
        predictions = self.amount_predictor.predict_with_confidence(X, model_type)

        return predictions

    def predict_payment_delay(
        self,
        invoice_data: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> pd.DataFrame:
        """Predict payment delays"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")

        suppliers_df = self.data_loader.load_suppliers()
        featured_df = self.feature_engineer.transform(invoice_data, suppliers_df)

        X, _ = self.delay_predictor.prepare_features(featured_df)
        predictions = self.delay_predictor.predict_payment_date(
            X, featured_df['due_date'], model_type
        )

        return predictions

    def predict_rejection_probability(
        self,
        invoice_data: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> pd.DataFrame:
        """Predict rejection probabilities"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")

        suppliers_df = self.data_loader.load_suppliers()
        featured_df = self.feature_engineer.transform(invoice_data, suppliers_df)

        X, _ = self.rejection_predictor.prepare_features(featured_df)
        predictions = self.rejection_predictor.predict_with_risk_level(X, model_type)

        return predictions
