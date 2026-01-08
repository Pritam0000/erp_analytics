# src/services/anomaly_service.py
"""Service layer for anomaly detection"""

import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime

from src.data.data_loader import InvoiceDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.anomaly.duplicate_detector import DuplicateDetector
from src.models.anomaly.amount_anomaly_detector import AmountAnomalyDetector
from src.models.anomaly.po_mismatch_detector import POMismatchDetector


class AnomalyService:
    """Orchestrate all anomaly detection"""

    def __init__(self, data_loader: Optional[InvoiceDataLoader] = None):
        """Initialize anomaly service"""
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader or InvoiceDataLoader(use_mock_db=True)

        # Initialize detectors
        self.duplicate_detector = DuplicateDetector()
        self.amount_detector = AmountAnomalyDetector()
        self.po_mismatch_detector = POMismatchDetector()
        self.feature_engineer = FeatureEngineer()

    def detect_all_anomalies(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all anomaly detection methods

        Args:
            start_date: Filter invoices from this date
            end_date: Filter invoices until this date

        Returns:
            Dictionary with all anomaly detection results
        """
        self.logger.info("Running all anomaly detections...")

        # Load data
        invoices_df = self.data_loader.load_invoices(start_date=start_date, end_date=end_date)
        suppliers_df = self.data_loader.load_suppliers()

        # Feature engineering for amount anomaly detection
        if not self.feature_engineer.is_fitted:
            self.feature_engineer.fit(invoices_df, suppliers_df)

        featured_df = self.feature_engineer.transform(invoices_df, suppliers_df)

        results = {}

        # Detect duplicates
        try:
            duplicates = self.duplicate_detector.detect_duplicates(invoices_df)
            results['duplicates'] = duplicates
            results['duplicate_report'] = self.duplicate_detector.get_duplicate_report(duplicates)
        except Exception as e:
            self.logger.error(f"Error detecting duplicates: {str(e)}")
            results['duplicates'] = pd.DataFrame()
            results['duplicate_report'] = {}

        # Detect amount anomalies
        try:
            amount_anomalies = self.amount_detector.detect_anomalies(featured_df)
            results['amount_anomalies'] = amount_anomalies
            results['amount_anomaly_report'] = self.amount_detector.get_anomaly_report(amount_anomalies)
        except Exception as e:
            self.logger.error(f"Error detecting amount anomalies: {str(e)}")
            results['amount_anomalies'] = pd.DataFrame()
            results['amount_anomaly_report'] = {}

        # Detect PO mismatches
        try:
            po_mismatches = self.po_mismatch_detector.detect_mismatches(featured_df)
            results['po_mismatches'] = po_mismatches
            results['po_mismatch_report'] = self.po_mismatch_detector.get_mismatch_report(po_mismatches)
            results['po_compliance'] = self.po_mismatch_detector.analyze_po_compliance(featured_df)
        except Exception as e:
            self.logger.error(f"Error detecting PO mismatches: {str(e)}")
            results['po_mismatches'] = pd.DataFrame()
            results['po_mismatch_report'] = {}
            results['po_compliance'] = {}

        self.logger.info("All anomaly detections complete")

        return results

    def detect_duplicates(self, invoices_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Detect duplicate invoices"""
        if invoices_df is None:
            invoices_df = self.data_loader.load_invoices()

        return self.duplicate_detector.detect_duplicates(invoices_df)

    def detect_amount_anomalies(
        self,
        invoices_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Detect amount anomalies"""
        if invoices_df is None:
            invoices_df = self.data_loader.load_invoices()

        suppliers_df = self.data_loader.load_suppliers()

        if not self.feature_engineer.is_fitted:
            self.feature_engineer.fit(invoices_df, suppliers_df)

        featured_df = self.feature_engineer.transform(invoices_df, suppliers_df)

        return self.amount_detector.detect_anomalies(featured_df)

    def detect_po_mismatches(
        self,
        invoices_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Detect PO mismatches"""
        if invoices_df is None:
            invoices_df = self.data_loader.load_invoices()

        suppliers_df = self.data_loader.load_suppliers()

        if not self.feature_engineer.is_fitted:
            self.feature_engineer.fit(invoices_df, suppliers_df)

        featured_df = self.feature_engineer.transform(invoices_df, suppliers_df)

        return self.po_mismatch_detector.detect_mismatches(featured_df)
