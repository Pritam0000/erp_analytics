# src/features/feature_engineer.py
"""Main feature engineering pipeline for invoice data"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import logging
import joblib
import os


class FeatureEngineer:
    """
    Transform raw invoice data into ML-ready features
    Handles temporal, financial, supplier, and risk features
    """

    def __init__(self):
        """Initialize feature engineer with transformers"""
        self.logger = logging.getLogger(__name__)

        # Transformers (will be fitted during training)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = None

        # Feature lists
        self.numerical_features = []
        self.categorical_features = []
        self.engineered_features = []

        # Fitted flag
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, suppliers_df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit transformers on training data

        Args:
            df: Invoice DataFrame
            suppliers_df: Supplier DataFrame

        Returns:
            self (fitted engineer)
        """
        self.logger.info("Fitting feature engineer...")

        # Create features
        df_featured = self._create_all_features(df, suppliers_df)

        # Identify feature types
        self.numerical_features = [
            'invoice_amount', 'tax_amount', 'payment_terms_days',
            'line_items_count', 'days_until_due', 'invoice_age_days',
            'po_invoice_variance', 'po_invoice_variance_abs',
            'tax_percentage', 'amount_per_line_item',
            'supplier_avg_amount', 'supplier_rejection_rate',
            'supplier_avg_delay', 'supplier_invoice_count'
        ]

        self.categorical_features = [
            'invoice_category', 'invoice_status', 'day_of_week',
            'month', 'quarter'
        ]

        # Fit scaler on numerical features
        numerical_cols = [col for col in self.numerical_features if col in df_featured.columns]
        if numerical_cols:
            self.scaler.fit(df_featured[numerical_cols].fillna(0))

        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in df_featured.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df_featured[col].fillna('UNKNOWN'))

        self.is_fitted = True
        self.logger.info("Feature engineer fitted successfully")

        return self

    def transform(
        self,
        df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
        include_target: bool = False
    ) -> pd.DataFrame:
        """
        Transform invoice data into engineered features

        Args:
            df: Invoice DataFrame
            suppliers_df: Supplier DataFrame
            include_target: Whether to include target variables

        Returns:
            DataFrame with engineered features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        self.logger.info(f"Transforming {len(df)} invoices...")

        # Create features
        df_featured = self._create_all_features(df, suppliers_df)

        # Transform numerical features
        numerical_cols = [col for col in self.numerical_features if col in df_featured.columns]
        if numerical_cols:
            df_featured[numerical_cols] = df_featured[numerical_cols].fillna(0)
            df_featured[numerical_cols] = self.scaler.transform(df_featured[numerical_cols])

        # Transform categorical features
        for col in self.categorical_features:
            if col in df_featured.columns:
                df_featured[col] = df_featured[col].fillna('UNKNOWN')
                df_featured[f'{col}_encoded'] = self.label_encoders[col].transform(df_featured[col])

        self.logger.info("Transformation complete")

        return df_featured

    def fit_transform(
        self,
        df: pd.DataFrame,
        suppliers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, suppliers_df)
        return self.transform(df, suppliers_df)

    def _create_all_features(
        self,
        df: pd.DataFrame,
        suppliers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create all engineered features"""
        df = df.copy()

        # Temporal features
        df = self._create_temporal_features(df)

        # Financial features
        df = self._create_financial_features(df)

        # Supplier features
        df = self._create_supplier_features(df, suppliers_df)

        # Risk indicators
        df = self._create_risk_indicators(df)

        # Target variables (if payment_date exists)
        if 'payment_date' in df.columns:
            df = self._create_target_variables(df)

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date/time based features"""
        # Ensure dates are datetime
        for col in ['invoice_date', 'due_date', 'payment_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Days until due
        df['days_until_due'] = (df['due_date'] - df['invoice_date']).dt.days

        # Invoice age (from today)
        df['invoice_age_days'] = (pd.Timestamp.now() - df['invoice_date']).dt.days

        # Temporal components
        df['day_of_week'] = df['invoice_date'].dt.dayofweek
        df['month'] = df['invoice_date'].dt.month
        df['quarter'] = df['invoice_date'].dt.quarter
        df['year'] = df['invoice_date'].dt.year

        return df

    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount and financial ratio features"""

        # PO variance
        df['po_invoice_variance'] = 0.0
        df['po_invoice_variance_abs'] = 0.0

        mask = (df['po_amount'].notna()) & (df['po_amount'] > 0)
        df.loc[mask, 'po_invoice_variance'] = (
            (df.loc[mask, 'invoice_amount'] - df.loc[mask, 'po_amount']) / df.loc[mask, 'po_amount']
        )
        df.loc[mask, 'po_invoice_variance_abs'] = (
            df.loc[mask, 'invoice_amount'] - df.loc[mask, 'po_amount']
        ).abs()

        # Tax percentage
        df['tax_percentage'] = df['tax_amount'] / df['invoice_amount'].replace(0, np.nan)
        df['tax_percentage'] = df['tax_percentage'].fillna(0)

        # Amount per line item
        df['amount_per_line_item'] = df['invoice_amount'] / df['line_items_count'].replace(0, 1)

        return df

    def _create_supplier_features(
        self,
        df: pd.DataFrame,
        suppliers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create supplier historical features"""

        # Merge supplier information
        supplier_features = suppliers_df[['supplier_id', 'avg_invoice_amount',
                                           'rejection_rate', 'avg_payment_delay_days',
                                           'total_invoices_count', 'risk_rating']]

        df = df.merge(
            supplier_features,
            on='supplier_id',
            how='left',
            suffixes=('', '_supplier')
        )

        # Rename for clarity
        df['supplier_avg_amount'] = df['avg_invoice_amount']
        df['supplier_rejection_rate'] = df['rejection_rate']
        df['supplier_avg_delay'] = df['avg_payment_delay_days']
        df['supplier_invoice_count'] = df['total_invoices_count']
        df['supplier_risk'] = df['risk_rating']

        # Drop original merged columns
        cols_to_drop = ['avg_invoice_amount', 'rejection_rate',
                        'avg_payment_delay_days', 'total_invoices_count']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        return df

    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary risk indicator features"""

        # High value invoice (>95th percentile)
        percentile_95 = df['invoice_amount'].quantile(0.95)
        df['is_high_value'] = (df['invoice_amount'] > percentile_95).astype(int)

        # Rush invoice (due in < 7 days)
        df['is_rush'] = (df['days_until_due'] < 7).astype(int)

        # Has PO
        df['has_po'] = df['po_number'].notna().astype(int)

        # PO mismatch (variance > 10%)
        df['po_mismatch_flag'] = (df['po_invoice_variance'].abs() > 0.1).astype(int)

        return df

    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""

        # Payment delay (actual - expected)
        mask = df['payment_date'].notna()
        df['payment_delay_days'] = 0
        df.loc[mask, 'payment_delay_days'] = (
            df.loc[mask, 'payment_date'] - df.loc[mask, 'due_date']
        ).dt.days

        # Is rejected
        df['is_rejected'] = (df['invoice_status'] == 'REJECTED').astype(int)

        return df

    def save(self, filepath: str):
        """Save fitted feature engineer"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        self.logger.info(f"Feature engineer saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'FeatureEngineer':
        """Load fitted feature engineer"""
        return joblib.load(filepath)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        features = self.numerical_features.copy()
        for cat in self.categorical_features:
            features.append(f'{cat}_encoded')
        return features
