# tests/analytics/anomaly_detection/test_transaction_monitor.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.anomaly_detection.transaction_monitor import TransactionMonitor

class TestTransactionMonitor(unittest.TestCase):
    """Test cases for TransactionMonitor class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.monitor = TransactionMonitor(Mock())
        # Mock database connection and logger
        self.monitor.logger = Mock()
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.start_date = self.test_date - timedelta(days=90)
        self.end_date = self.test_date
        
        # Set up mock GL data with realistic transaction patterns
        self.mock_gl_data = self.generate_mock_gl_data()
        self.mock_ap_data = self.generate_mock_ap_data()
        self.mock_journal_data = self.generate_mock_journal_data()

    def generate_mock_gl_data(self) -> pd.DataFrame:
        """Generate mock GL transaction data"""
        # Create base data
        n_transactions = 1000
        data = pd.DataFrame({
            'journal_id': range(1, n_transactions + 1),
            'journal_date': pd.date_range(start='2023-10-01', end='2024-01-01', freq='H')[:n_transactions],
            'line_id': range(1, n_transactions + 1),
            'debit_amount': np.random.uniform(1000, 10000, size=n_transactions),
            'credit_amount': np.random.uniform(1000, 10000, size=n_transactions),
            'account_id': np.random.choice(range(1001, 1011), size=n_transactions)
        })
        
        # Add anomalous transactions
        anomaly_indices = np.random.choice(n_transactions, size=20, replace=False)
        
        # Volume anomalies
        data.loc[anomaly_indices[:5], 'debit_amount'] *= 5
        data.loc[anomaly_indices[:5], 'credit_amount'] *= 5
        
        # Timing anomalies (unusual hours)
        data.loc[anomaly_indices[5:10], 'journal_date'] = data.loc[
            anomaly_indices[5:10], 'journal_date'
        ].apply(lambda x: x.replace(hour=2))  # Late night transactions
        
        # Pattern anomalies
        data.loc[anomaly_indices[10:], 'account_id'] = 9999  # Unusual account
        
        return data

    def generate_mock_ap_data(self) -> pd.DataFrame:
        """Generate mock AP transaction data"""
        n_transactions = 500
        data = pd.DataFrame({
            'invoice_id': range(1, n_transactions + 1),
            'supplier_id': np.random.choice(range(101, 111), size=n_transactions),
            'invoice_date': pd.date_range(start='2023-10-01', end='2024-01-01', freq='D')[:n_transactions],
            'due_date': pd.date_range(start='2023-11-01', end='2024-02-01', freq='D')[:n_transactions],
            'amount': np.random.uniform(5000, 50000, size=n_transactions),
            'payment_date': None,
            'payment_method': np.random.choice(['CHECK', 'ACH', 'WIRE'], size=n_transactions)
        })
        
        # Add payment dates for some invoices
        paid_mask = np.random.choice([True, False], size=n_transactions, p=[0.8, 0.2])
        data.loc[paid_mask, 'payment_date'] = data.loc[paid_mask, 'due_date'] + pd.Timedelta(days=np.random.randint(-5, 15))
        
        return data

    def generate_mock_journal_data(self) -> pd.DataFrame:
        """Generate mock journal entry data"""
        n_journals = 300
        data = pd.DataFrame({
            'journal_id': range(1, n_journals + 1),
            'journal_date': pd.date_range(start='2023-10-01', end='2024-01-01', freq='D')[:n_journals],
            'journal_number': [f'JE{i:05d}' for i in range(1, n_journals + 1)],
            'batch_number': np.random.choice(range(1, 11), size=n_journals),
            'status': 'POSTED',
            'period_name': [f'2023-{i:02d}' for i in np.random.randint(10, 13, size=n_journals)]
        })
        
        # Add some anomalous journals
        anomaly_indices = np.random.choice(n_journals, size=10, replace=False)
        data.loc[anomaly_indices, 'journal_number'] = 'MANUAL_JE'
        
        return data

    def test_prepare_monitoring_data(self):
        """Test preparation of monitoring data"""
        # Mock data loading functions
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.monitor._get_journal_data = Mock(return_value=self.mock_journal_data)
        
        # Test data preparation
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        # Verify structure
        self.assertIsInstance(monitoring_data, pd.DataFrame)
        
        # Verify feature categories
        expected_categories = [
            'amount_', 'volume_', 'payment_', 
            'interaction_', 'timing_'
        ]
        
        for category in expected_categories:
            self.assertTrue(
                any(col.startswith(category) for col in monitoring_data.columns),
                f"Missing features for category: {category}"
            )
        
        # Verify key features
        key_features = [
            'total_debits', 'total_credits', 'transaction_intensity',
            'unique_accounts', 'payment_delay', 'nonbusiness_hour_ratio'
        ]
        
        for feature in key_features:
            self.assertIn(feature, monitoring_data.columns)

    def test_analyze_patterns(self):
        """Test pattern analysis"""
        # Mock data loading functions
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.monitor._get_journal_data = Mock(return_value=self.mock_journal_data)
        
        # Test pattern analysis
        pattern_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        results = self.monitor.analyze_payment_patterns(pattern_data)
        
        # Verify structure
        self.assertIn('cluster_metrics', results)
        self.assertIn('anomaly_metrics', results)
        self.assertIn('pattern_profiles', results)
        
        # Verify metrics
        clusters = results['cluster_metrics']
        self.assertIn('n_clusters', clusters)
        self.assertIn('cluster_sizes', clusters)
        self.assertIn('cluster_stats', clusters)
        
        # Verify anomaly detection
        anomalies = results['anomaly_metrics']
        self.assertIn('anomaly_count', anomalies)
        self.assertIn('anomaly_rate', anomalies)
        self.assertIn('anomaly_scores', anomalies)

    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Mock data loading functions
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.monitor._get_journal_data = Mock(return_value=self.mock_journal_data)
        
        # Prepare and analyze data
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        self.monitor.train_detector(monitoring_data)
        
        # Test detection
        detection_results = self.monitor.detect_anomalies(monitoring_data)
        
        # Verify structure
        self.assertIn('anomalies', detection_results)
        self.assertIn('detection_summary', detection_results)
        
        # Verify anomaly details
        anomalies = detection_results['anomalies']
        self.assertGreater(len(anomalies), 0)
        
        for anomaly in anomalies:
            self.assertIn('transaction_id', anomaly)
            self.assertIn('transaction_date', anomaly)
            self.assertIn('anomaly_score', anomaly)
            self.assertIn('risk_level', anomaly)
            self.assertIn('anomaly_factors', anomaly)

    def test_calculate_anomaly_scores(self):
        """Test anomaly score calculation"""
        test_data = pd.DataFrame({
            'amount_zscore': np.random.normal(0, 1, size=100),
            'volume_zscore': np.random.normal(0, 1, size=100),
            'timing_irregularity': np.random.uniform(0, 1, size=100)
        })
        
        # Add known anomalies
        test_data.loc[0:4, 'amount_zscore'] = 5.0  # Clear amount anomalies
        test_data.loc[5:9, 'volume_zscore'] = 5.0  # Clear volume anomalies
        
        # Calculate scores
        scores = self.monitor._calculate_anomaly_scores(
            test_data.values,  # Scaled features
            np.ones(len(test_data)) * -1,  # Isolation Forest predictions
            np.zeros(len(test_data))  # Ensemble predictions
        )
        
        # Verify scores
        self.assertEqual(len(scores), len(test_data))
        self.assertTrue(all(0 <= score <= 1 for score in scores['composite_score']))
        
        # Verify anomaly identification
        high_scores = scores[scores['composite_score'] > scores['composite_score'].mean()]
        self.assertGreaterEqual(len(high_scores), 10)  # Should detect our injected anomalies

    def test_analyze_detection_patterns(self):
        """Test analysis of detection patterns"""
        # Mock data loading functions
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.monitor._get_journal_data = Mock(return_value=self.mock_journal_data)
        
        # Prepare data and train detector
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        self.monitor.train_detector(monitoring_data)
        detection_results = self.monitor.detect_anomalies(monitoring_data)
        
        # Analyze patterns
        pattern_analysis = self.monitor.analyze_detection_patterns(
            self.start_date,
            self.end_date
        )
        
        # Verify structure
        self.assertIn('temporal_patterns', pattern_analysis)
        self.assertIn('factor_patterns', pattern_analysis)
        self.assertIn('risk_distribution', pattern_analysis)
        
        # Verify pattern details
        temporal = pattern_analysis['temporal_patterns']
        self.assertIn('daily_distribution', temporal)
        self.assertIn('hourly_distribution', temporal)
        self.assertIn('temporal_metrics', temporal)

    def test_save_and_load_detector(self):
        """Test saving and loading detector state"""
        # Prepare data and train detector
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.monitor._get_journal_data = Mock(return_value=self.mock_journal_data)
        
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        self.monitor.train_detector(monitoring_data)
        
        # Save detector state
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'detector_state.joblib')
            self.monitor.save_detector(save_path)
            
            # Create new monitor and load state
            new_monitor = TransactionMonitor(Mock())
            new_monitor.load_detector(save_path)
            
            # Verify loaded state
            self.assertEqual(
                self.monitor.feature_columns,
                new_monitor.feature_columns
            )
            self.assertEqual(
                len(self.monitor.model_metrics),
                len(new_monitor.model_metrics)
            )

if __name__ == '__main__':
    unittest.main()