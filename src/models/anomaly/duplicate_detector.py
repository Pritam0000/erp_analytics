# src/models/anomaly/duplicate_detector.py
"""Detect duplicate invoices using Isolation Forest and fuzzy matching"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
import logging
from datetime import timedelta

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available. Using basic string matching.")

from config.model_config import ISOLATION_FOREST_CONFIG
from config.feature_config import ANOMALY_THRESHOLDS


class DuplicateDetector:
    """
    Detect duplicate invoices using multiple methods:
    1. Exact duplicates
    2. Fuzzy matching (similar invoice numbers, amounts, dates)
    3. Isolation Forest anomaly detection
    """

    def __init__(self):
        """Initialize duplicate detector"""
        self.logger = logging.getLogger(__name__)
        self.isolation_forest = IsolationForest(**ISOLATION_FOREST_CONFIG)
        self.is_fitted = False

    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all types of duplicates

        Args:
            df: DataFrame with invoice data

        Returns:
            DataFrame with duplicate information
        """
        self.logger.info(f"Detecting duplicates in {len(df)} invoices...")

        # Find different types of duplicates
        exact_dups = self._find_exact_duplicates(df)
        fuzzy_dups = self._find_fuzzy_duplicates(df)
        ml_anomalies = self._find_ml_anomalies(df)

        # Combine results
        all_duplicates = self._combine_duplicate_results(
            df, exact_dups, fuzzy_dups, ml_anomalies
        )

        self.logger.info(f"Found {len(all_duplicates)} potential duplicates")

        return all_duplicates

    def _find_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find exact duplicate invoices"""
        # Group by key fields
        duplicate_mask = df.duplicated(
            subset=['invoice_number', 'supplier_id', 'invoice_amount'],
            keep=False
        )

        exact_dups = df[duplicate_mask].copy()
        exact_dups['duplicate_type'] = 'EXACT'
        exact_dups['duplicate_score'] = 1.0

        self.logger.info(f"Found {len(exact_dups)} exact duplicates")

        return exact_dups

    def _find_fuzzy_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find fuzzy/near duplicates"""
        fuzzy_dups = []

        # Convert dates to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['invoice_date']):
            df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')

        # Group by supplier to reduce comparison space
        for supplier_id, group in df.groupby('supplier_id'):
            if len(group) < 2:
                continue

            # Compare within supplier
            for i, row1 in group.iterrows():
                for j, row2 in group.iterrows():
                    if i >= j:  # Avoid duplicate comparisons
                        continue

                    similarity = self._calculate_similarity(row1, row2)

                    if similarity > ANOMALY_THRESHOLDS['duplicate_similarity']:
                        fuzzy_dups.append({
                            'invoice_id': row1['invoice_id'],
                            'invoice_number': row1['invoice_number'],
                            'supplier_id': row1['supplier_id'],
                            'invoice_amount': row1['invoice_amount'],
                            'invoice_date': row1['invoice_date'],
                            'duplicate_type': 'FUZZY',
                            'duplicate_score': similarity,
                            'matched_with_id': row2['invoice_id'],
                            'matched_with_number': row2['invoice_number']
                        })

        if fuzzy_dups:
            fuzzy_df = pd.DataFrame(fuzzy_dups)
            self.logger.info(f"Found {len(fuzzy_df)} fuzzy duplicates")
            return fuzzy_df
        else:
            return pd.DataFrame()

    def _calculate_similarity(self, row1: pd.Series, row2: pd.Series) -> float:
        """
        Calculate similarity score between two invoices

        Returns:
            Float between 0 and 1 (1 = identical)
        """
        score = 0.0

        # Invoice number similarity (40% weight)
        if FUZZYWUZZY_AVAILABLE:
            inv_num_sim = fuzz.ratio(str(row1['invoice_number']), str(row2['invoice_number'])) / 100
        else:
            inv_num_sim = 1.0 if row1['invoice_number'] == row2['invoice_number'] else 0.0
        score += inv_num_sim * 0.4

        # Amount similarity (30% weight)
        amt_diff = abs(row1['invoice_amount'] - row2['invoice_amount'])
        avg_amt = (row1['invoice_amount'] + row2['invoice_amount']) / 2
        amt_sim = 1 - min(amt_diff / avg_amt if avg_amt > 0 else 0, 1)
        score += amt_sim * 0.3

        # Date similarity (20% weight)
        date_diff = abs((row1['invoice_date'] - row2['invoice_date']).days)
        date_sim = 1 - min(date_diff / 30, 1)  # 30 days = 0 similarity
        score += date_sim * 0.2

        # Supplier match (10% weight)
        if row1['supplier_id'] == row2['supplier_id']:
            score += 0.1

        return score

    def _find_ml_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use Isolation Forest to find anomalous patterns"""
        # Prepare features for anomaly detection
        features = []
        feature_cols = ['invoice_amount', 'supplier_id', 'line_items_count']

        # Create numerical features
        X = df[feature_cols].copy()

        # Convert date to numeric (days since earliest invoice)
        if pd.api.types.is_datetime64_any_dtype(df['invoice_date']):
            min_date = df['invoice_date'].min()
            X['days_since_start'] = (df['invoice_date'] - min_date).dt.days
        else:
            X['days_since_start'] = 0

        # Fill NaN
        X = X.fillna(0)

        # Fit and predict
        self.isolation_forest.fit(X)
        predictions = self.isolation_forest.predict(X)
        anomaly_scores = self.isolation_forest.score_samples(X)

        # -1 indicates anomaly
        anomaly_mask = predictions == -1

        ml_anomalies = df[anomaly_mask].copy()
        ml_anomalies['duplicate_type'] = 'ML_ANOMALY'
        ml_anomalies['duplicate_score'] = -anomaly_scores[anomaly_mask]  # Convert to positive

        self.logger.info(f"Found {len(ml_anomalies)} ML-detected anomalies")
        self.is_fitted = True

        return ml_anomalies

    def _combine_duplicate_results(
        self,
        df: pd.DataFrame,
        exact_dups: pd.DataFrame,
        fuzzy_dups: pd.DataFrame,
        ml_anomalies: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine all duplicate detection results"""

        all_dups = []

        # Add exact duplicates
        if len(exact_dups) > 0:
            all_dups.append(exact_dups[[
                'invoice_id', 'invoice_number', 'supplier_id',
                'invoice_amount', 'invoice_date', 'duplicate_type', 'duplicate_score'
            ]])

        # Add fuzzy duplicates
        if len(fuzzy_dups) > 0:
            fuzzy_cols = ['invoice_id', 'invoice_number', 'supplier_id',
                          'invoice_amount', 'invoice_date', 'duplicate_type', 'duplicate_score']
            if 'matched_with_id' in fuzzy_dups.columns:
                fuzzy_cols.extend(['matched_with_id', 'matched_with_number'])

            all_dups.append(fuzzy_dups[fuzzy_cols])

        # Add ML anomalies
        if len(ml_anomalies) > 0:
            all_dups.append(ml_anomalies[[
                'invoice_id', 'invoice_number', 'supplier_id',
                'invoice_amount', 'invoice_date', 'duplicate_type', 'duplicate_score'
            ]])

        if all_dups:
            combined = pd.concat(all_dups, ignore_index=True)

            # Remove actual duplicates in results (keep highest score)
            combined = combined.sort_values('duplicate_score', ascending=False)
            combined = combined.drop_duplicates(subset=['invoice_id'], keep='first')

            # Add severity
            combined['severity'] = pd.cut(
                combined['duplicate_score'],
                bins=[-0.01, 0.5, 0.8, 1.01],
                labels=['Low', 'Medium', 'High']
            )

            return combined.sort_values('duplicate_score', ascending=False)
        else:
            return pd.DataFrame()

    def get_duplicate_report(self, duplicates_df: pd.DataFrame) -> Dict:
        """Generate summary report of duplicates"""
        if len(duplicates_df) == 0:
            return {
                'total_duplicates': 0,
                'by_type': {},
                'by_severity': {},
                'total_amount_at_risk': 0
            }

        report = {
            'total_duplicates': len(duplicates_df),
            'by_type': duplicates_df['duplicate_type'].value_counts().to_dict(),
            'by_severity': duplicates_df['severity'].value_counts().to_dict(),
            'total_amount_at_risk': duplicates_df['invoice_amount'].sum(),
            'avg_duplicate_score': duplicates_df['duplicate_score'].mean(),
            'top_suppliers_affected': duplicates_df['supplier_id'].value_counts().head(5).to_dict()
        }

        return report
