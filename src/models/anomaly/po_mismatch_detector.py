# src/models/anomaly/po_mismatch_detector.py
"""Detect PO-Invoice mismatches using rule-based and ML approaches"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from config.feature_config import ANOMALY_THRESHOLDS


class POMismatchDetector:
    """
    Detect mismatches between Purchase Orders and Invoices
    Uses rule-based approach with configurable thresholds
    """

    def __init__(self, variance_threshold: float = 0.10):
        """
        Initialize PO mismatch detector

        Args:
            variance_threshold: Acceptable variance percentage (default 10%)
        """
        self.logger = logging.getLogger(__name__)
        self.variance_threshold = variance_threshold

    def detect_mismatches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect PO-invoice mismatches

        Args:
            df: DataFrame with invoice and PO data

        Returns:
            DataFrame with mismatch information
        """
        self.logger.info(f"Detecting PO mismatches in {len(df)} invoices...")

        # Filter invoices that should have POs
        df_with_po = df[df['po_number'].notna() & (df['po_amount'].notna())].copy()

        if len(df_with_po) == 0:
            self.logger.warning("No invoices with PO information found")
            return pd.DataFrame()

        self.logger.info(f"Analyzing {len(df_with_po)} invoices with POs")

        # Calculate variances
        df_with_po = self._calculate_variances(df_with_po)

        # Detect different types of mismatches
        amount_mismatches = self._detect_amount_mismatches(df_with_po)
        missing_po_mismatches = self._detect_missing_po(df)
        excessive_variance = self._detect_excessive_variance(df_with_po)

        # Combine results
        all_mismatches = self._combine_mismatch_results(
            amount_mismatches, missing_po_mismatches, excessive_variance
        )

        self.logger.info(f"Found {len(all_mismatches)} PO mismatches")

        return all_mismatches

    def _calculate_variances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PO-invoice variances"""
        # Absolute variance
        df['po_variance_amount'] = df['invoice_amount'] - df['po_amount']

        # Percentage variance
        df['po_variance_percent'] = (
            df['po_variance_amount'] / df['po_amount'].replace(0, np.nan)
        )

        # Absolute percentage (for comparison)
        df['po_variance_percent_abs'] = df['po_variance_percent'].abs()

        return df

    def _detect_amount_mismatches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect invoices where amount differs from PO beyond threshold"""
        # Mismatch if variance > threshold
        mismatch_mask = df['po_variance_percent_abs'] > self.variance_threshold

        mismatches = df[mismatch_mask].copy()
        mismatches['mismatch_type'] = 'AMOUNT_VARIANCE'

        # Calculate severity based on variance magnitude
        mismatches['mismatch_score'] = mismatches['po_variance_percent_abs'] / self.variance_threshold

        self.logger.info(f"Found {len(mismatches)} amount variance mismatches")

        return mismatches

    def _detect_missing_po(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect invoices that should have PO but don't"""
        # High-value invoices without PO (business rule)
        high_value_threshold = df['invoice_amount'].quantile(0.75)  # 75th percentile

        missing_po_mask = (
            (df['po_number'].isna() | (df['po_amount'].isna())) &
            (df['invoice_amount'] > high_value_threshold)
        )

        mismatches = df[missing_po_mask].copy()
        mismatches['mismatch_type'] = 'MISSING_PO'
        mismatches['po_variance_amount'] = 0
        mismatches['po_variance_percent'] = 0
        mismatches['po_variance_percent_abs'] = 0

        # Score based on how much over threshold
        mismatches['mismatch_score'] = (
            mismatches['invoice_amount'] / high_value_threshold
        )

        self.logger.info(f"Found {len(mismatches)} missing PO mismatches")

        return mismatches

    def _detect_excessive_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect invoices with excessive (>50%) variance"""
        excessive_threshold = 0.50  # 50%

        excessive_mask = df['po_variance_percent_abs'] > excessive_threshold

        mismatches = df[excessive_mask].copy()
        mismatches['mismatch_type'] = 'EXCESSIVE_VARIANCE'
        mismatches['mismatch_score'] = mismatches['po_variance_percent_abs']

        self.logger.info(f"Found {len(mismatches)} excessive variance mismatches")

        return mismatches

    def _combine_mismatch_results(
        self,
        amount_mismatches: pd.DataFrame,
        missing_po: pd.DataFrame,
        excessive_variance: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine all mismatch detection results"""
        all_mismatches = []

        # Standard columns to keep
        result_cols = [
            'invoice_id', 'invoice_number', 'supplier_id', 'supplier_name',
            'invoice_amount', 'po_number', 'po_amount',
            'mismatch_type', 'mismatch_score'
        ]

        # Add optional columns if they exist
        optional_cols = ['po_variance_amount', 'po_variance_percent', 'po_variance_percent_abs']

        for mismatch_df in [amount_mismatches, missing_po, excessive_variance]:
            if len(mismatch_df) > 0:
                # Select columns that exist
                cols_to_keep = [col for col in result_cols if col in mismatch_df.columns]
                cols_to_keep.extend([col for col in optional_cols if col in mismatch_df.columns])

                all_mismatches.append(mismatch_df[cols_to_keep])

        if not all_mismatches:
            return pd.DataFrame()

        # Concatenate all results
        combined = pd.concat(all_mismatches, ignore_index=True)

        # Remove duplicates (keep highest score)
        combined = combined.sort_values('mismatch_score', ascending=False)
        combined = combined.drop_duplicates(subset=['invoice_id'], keep='first')

        # Add severity levels
        combined['severity'] = pd.cut(
            combined['mismatch_score'],
            bins=[-0.01, 0.2, 0.5, 10.0],
            labels=['Low', 'Medium', 'High']
        )

        # Add recommendation
        combined['recommendation'] = combined['mismatch_type'].map({
            'AMOUNT_VARIANCE': 'Review with procurement',
            'MISSING_PO': 'Request PO before payment',
            'EXCESSIVE_VARIANCE': 'Investigate immediately'
        })

        # Sort by severity and score
        combined = combined.sort_values(['severity', 'mismatch_score'], ascending=[False, False])

        return combined

    def get_mismatch_report(self, mismatches_df: pd.DataFrame) -> Dict:
        """Generate summary report of PO mismatches"""
        if len(mismatches_df) == 0:
            return {
                'total_mismatches': 0,
                'by_type': {},
                'by_severity': {},
                'total_amount_at_risk': 0
            }

        report = {
            'total_mismatches': len(mismatches_df),
            'by_type': mismatches_df['mismatch_type'].value_counts().to_dict(),
            'by_severity': mismatches_df['severity'].value_counts().to_dict(),
            'total_amount_at_risk': mismatches_df['invoice_amount'].sum(),
            'avg_mismatch_score': mismatches_df['mismatch_score'].mean(),
            'total_variance_amount': mismatches_df.get('po_variance_amount', pd.Series([0])).abs().sum(),
            'top_suppliers_affected': mismatches_df['supplier_id'].value_counts().head(5).to_dict()
        }

        # Add variance statistics if available
        if 'po_variance_percent' in mismatches_df.columns:
            variance_pcts = mismatches_df['po_variance_percent'].dropna()
            if len(variance_pcts) > 0:
                report['avg_variance_percent'] = variance_pcts.abs().mean()
                report['max_variance_percent'] = variance_pcts.abs().max()

        return report

    def analyze_po_compliance(self, df: pd.DataFrame) -> Dict:
        """Analyze overall PO compliance"""
        total_invoices = len(df)

        # High value threshold (75th percentile)
        high_value_threshold = df['invoice_amount'].quantile(0.75)
        high_value_invoices = df[df['invoice_amount'] > high_value_threshold]

        # Invoices with PO
        with_po = df[df['po_number'].notna() & (df['po_amount'].notna())]

        # High value invoices with PO
        high_value_with_po = high_value_invoices[
            high_value_invoices['po_number'].notna() &
            (high_value_invoices['po_amount'].notna())
        ]

        compliance = {
            'total_invoices': total_invoices,
            'invoices_with_po': len(with_po),
            'po_coverage_rate': len(with_po) / total_invoices if total_invoices > 0 else 0,
            'high_value_invoices': len(high_value_invoices),
            'high_value_with_po': len(high_value_with_po),
            'high_value_po_rate': (
                len(high_value_with_po) / len(high_value_invoices)
                if len(high_value_invoices) > 0 else 0
            ),
            'high_value_threshold': high_value_threshold
        }

        return compliance
