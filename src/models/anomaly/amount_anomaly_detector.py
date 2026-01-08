# src/models/anomaly/amount_anomaly_detector.py
"""Detect unusual invoice amounts using LOF and Isolation Forest"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

from config.model_config import LOF_CONFIG, ISOLATION_FOREST_CONFIG
from config.feature_config import ANOMALY_THRESHOLDS


class AmountAnomalyDetector:
    """
    Detect unusual invoice amounts using:
    1. Local Outlier Factor (LOF)
    2. Isolation Forest
    3. Statistical methods (Z-score)
    4. Business rules
    """

    def __init__(self):
        """Initialize amount anomaly detector"""
        self.logger = logging.getLogger(__name__)
        self.lof = LocalOutlierFactor(**LOF_CONFIG)
        self.isolation_forest = IsolationForest(**ISOLATION_FOREST_CONFIG)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect amount anomalies using ensemble approach

        Args:
            df: DataFrame with invoice data

        Returns:
            DataFrame with anomaly information
        """
        self.logger.info(f"Detecting amount anomalies in {len(df)} invoices...")

        # Prepare features
        X, feature_df = self._prepare_features(df)

        # Detect using different methods
        lof_anomalies = self._detect_with_lof(X, feature_df)
        if_anomalies = self._detect_with_isolation_forest(X, feature_df)
        stat_anomalies = self._detect_statistical_anomalies(df)
        business_anomalies = self._detect_business_rule_anomalies(df)

        # Combine results
        all_anomalies = self._combine_anomaly_results(
            df, lof_anomalies, if_anomalies, stat_anomalies, business_anomalies
        )

        self.logger.info(f"Found {len(all_anomalies)} amount anomalies")

        return all_anomalies

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare features for anomaly detection"""
        feature_cols = []

        # Amount features
        feature_df = pd.DataFrame()
        feature_df['invoice_amount'] = df['invoice_amount']

        if 'line_items_count' in df.columns:
            feature_df['amount_per_line'] = df['invoice_amount'] / df['line_items_count'].replace(0, 1)
            feature_cols.append('amount_per_line')

        # PO variance
        if 'po_amount' in df.columns and 'invoice_amount' in df.columns:
            mask = df['po_amount'].notna() & (df['po_amount'] > 0)
            feature_df['po_variance'] = 0.0
            feature_df.loc[mask, 'po_variance'] = (
                (df.loc[mask, 'invoice_amount'] - df.loc[mask, 'po_amount']).abs()
            )
            feature_cols.append('po_variance')

        # Supplier average
        if 'supplier_avg_amount' in df.columns:
            feature_df['vs_supplier_avg'] = df['invoice_amount'] / df['supplier_avg_amount'].replace(0, 1)
            feature_cols.append('vs_supplier_avg')

        # Tax percentage
        if 'tax_amount' in df.columns:
            feature_df['tax_pct'] = df['tax_amount'] / df['invoice_amount'].replace(0, 1)
            feature_cols.append('tax_pct')

        feature_cols = ['invoice_amount'] + feature_cols
        X = feature_df[feature_cols].fillna(0).values

        # Scale features
        X = self.scaler.fit_transform(X)

        return X, feature_df

    def _detect_with_lof(self, X: np.ndarray, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Local Outlier Factor"""
        # LOF returns -1 for outliers
        predictions = self.lof.fit_predict(X)
        outlier_scores = -self.lof.negative_outlier_factor_  # Convert to positive

        anomaly_mask = predictions == -1
        anomalies = feature_df[anomaly_mask].copy()
        anomalies['detection_method'] = 'LOF'
        anomalies['anomaly_score'] = outlier_scores[anomaly_mask]

        self.logger.info(f"LOF detected {len(anomalies)} anomalies")

        return anomalies

    def _detect_with_isolation_forest(
        self,
        X: np.ndarray,
        feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest"""
        self.isolation_forest.fit(X)
        predictions = self.isolation_forest.predict(X)
        anomaly_scores = -self.isolation_forest.score_samples(X)  # Convert to positive

        anomaly_mask = predictions == -1
        anomalies = feature_df[anomaly_mask].copy()
        anomalies['detection_method'] = 'IsolationForest'
        anomalies['anomaly_score'] = anomaly_scores[anomaly_mask]

        self.logger.info(f"Isolation Forest detected {len(anomalies)} anomalies")
        self.is_fitted = True

        return anomalies

    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Z-score"""
        amounts = df['invoice_amount']

        # Calculate Z-scores
        mean_amt = amounts.mean()
        std_amt = amounts.std()
        z_scores = (amounts - mean_amt) / std_amt

        # Flag if |Z-score| > 3
        anomaly_mask = z_scores.abs() > 3

        anomalies = df[anomaly_mask].copy()
        anomalies['detection_method'] = 'Statistical'
        anomalies['anomaly_score'] = z_scores[anomaly_mask].abs() / 3  # Normalize to 0-1

        self.logger.info(f"Statistical method detected {len(anomalies)} anomalies")

        return anomalies

    def _detect_business_rule_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using business rules"""
        anomalies_list = []

        # Rule 1: Amount > 10x supplier average
        if 'supplier_avg_amount' in df.columns:
            mask1 = df['invoice_amount'] > (df['supplier_avg_amount'] * 10)
            if mask1.any():
                rule1_anomalies = df[mask1].copy()
                rule1_anomalies['detection_method'] = 'BusinessRule_10xSupplierAvg'
                rule1_anomalies['anomaly_score'] = (
                    df.loc[mask1, 'invoice_amount'] / df.loc[mask1, 'supplier_avg_amount']
                ) / 10
                anomalies_list.append(rule1_anomalies)

        # Rule 2: PO variance > 50%
        if 'po_amount' in df.columns and 'invoice_amount' in df.columns:
            mask2 = df['po_amount'].notna() & (df['po_amount'] > 0)
            po_variance = (df['invoice_amount'] - df['po_amount']).abs() / df['po_amount']
            mask2 = mask2 & (po_variance > 0.5)

            if mask2.any():
                rule2_anomalies = df[mask2].copy()
                rule2_anomalies['detection_method'] = 'BusinessRule_POVariance50pct'
                rule2_anomalies['anomaly_score'] = po_variance[mask2]
                anomalies_list.append(rule2_anomalies)

        # Rule 3: Amount > 99th percentile
        percentile_99 = df['invoice_amount'].quantile(0.99)
        mask3 = df['invoice_amount'] > percentile_99
        if mask3.any():
            rule3_anomalies = df[mask3].copy()
            rule3_anomalies['detection_method'] = 'BusinessRule_99thPercentile'
            rule3_anomalies['anomaly_score'] = (
                df.loc[mask3, 'invoice_amount'] / percentile_99
            )
            anomalies_list.append(rule3_anomalies)

        if anomalies_list:
            business_anomalies = pd.concat(anomalies_list, ignore_index=True)
            self.logger.info(f"Business rules detected {len(business_anomalies)} anomalies")
            return business_anomalies
        else:
            return pd.DataFrame()

    def _combine_anomaly_results(
        self,
        df: pd.DataFrame,
        lof_anomalies: pd.DataFrame,
        if_anomalies: pd.DataFrame,
        stat_anomalies: pd.DataFrame,
        business_anomalies: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine results from all detection methods"""

        # Collect all anomalous invoice IDs
        all_anomaly_ids = set()

        for anom_df in [lof_anomalies, if_anomalies, stat_anomalies, business_anomalies]:
            if len(anom_df) > 0 and 'invoice_id' in anom_df.columns:
                all_anomaly_ids.update(anom_df['invoice_id'].values)

        if not all_anomaly_ids:
            return pd.DataFrame()

        # Create results DataFrame
        results = []

        for invoice_id in all_anomaly_ids:
            # Get invoice data
            invoice_row = df[df['invoice_id'] == invoice_id].iloc[0]

            # Collect detection methods and scores
            detection_methods = []
            scores = []

            for anom_df in [lof_anomalies, if_anomalies, stat_anomalies, business_anomalies]:
                if len(anom_df) > 0 and invoice_id in anom_df['invoice_id'].values:
                    method_row = anom_df[anom_df['invoice_id'] == invoice_id].iloc[0]
                    detection_methods.append(method_row['detection_method'])
                    scores.append(method_row['anomaly_score'])

            # Calculate ensemble score
            ensemble_score = np.mean(scores) if scores else 0

            results.append({
                'invoice_id': invoice_id,
                'invoice_number': invoice_row.get('invoice_number', ''),
                'supplier_id': invoice_row.get('supplier_id', ''),
                'supplier_name': invoice_row.get('supplier_name', ''),
                'invoice_amount': invoice_row.get('invoice_amount', 0),
                'supplier_avg_amount': invoice_row.get('supplier_avg_amount', 0),
                'detection_methods': ', '.join(detection_methods),
                'num_methods_flagged': len(detection_methods),
                'anomaly_score': ensemble_score
            })

        results_df = pd.DataFrame(results)

        # Add severity
        results_df['severity'] = pd.cut(
            results_df['anomaly_score'],
            bins=[-0.01, 0.5, 0.75, 2.0],
            labels=['Low', 'Medium', 'High']
        )

        # Sort by score
        results_df = results_df.sort_values('anomaly_score', ascending=False)

        return results_df

    def get_anomaly_report(self, anomalies_df: pd.DataFrame) -> Dict:
        """Generate summary report of amount anomalies"""
        if len(anomalies_df) == 0:
            return {
                'total_anomalies': 0,
                'by_severity': {},
                'total_amount_flagged': 0
            }

        report = {
            'total_anomalies': len(anomalies_df),
            'by_severity': anomalies_df['severity'].value_counts().to_dict(),
            'total_amount_flagged': anomalies_df['invoice_amount'].sum(),
            'avg_anomaly_score': anomalies_df['anomaly_score'].mean(),
            'max_anomaly_amount': anomalies_df['invoice_amount'].max(),
            'top_suppliers_affected': anomalies_df['supplier_id'].value_counts().head(5).to_dict()
        }

        return report
