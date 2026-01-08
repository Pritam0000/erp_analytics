# tests/analytics/anomaly_detection/test_account_monitor.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.anomaly_detection.account_monitor import AccountMonitor

class TestAccountMonitor(unittest.TestCase):
    """Test cases for AccountMonitor class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.monitor = AccountMonitor(Mock())
        # Mock database connection and logger
        self.monitor.logger = Mock()
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.start_date = self.test_date - timedelta(days=90)
        self.end_date = self.test_date
        
        # Set up mock GL data
        self.mock_gl_data = pd.DataFrame({
            'journal_id': range(1, 101),
            'journal_date': pd.date_range(start='2023-10-01', end='2024-01-01', freq='D')[:100],
            'line_id': range(1, 101),
            'account_id': np.repeat([1001, 1002, 1003, 1004], 25),
            'debit_amount': np.random.uniform(1000, 10000, size=100),
            'credit_amount': np.random.uniform(1000, 10000, size=100)
        })
        
        # Add some anomalies for testing
        anomaly_indices = np.random.choice(100, size=5, replace=False)
        self.mock_gl_data.loc[anomaly_indices, 'debit_amount'] *= 5
        self.mock_gl_data.loc[anomaly_indices, 'credit_amount'] *= 5
        
        # Set up mock balance data
        dates = pd.date_range(start='2023-10-01', end='2024-01-01', freq='D')
        self.mock_balance_data = pd.DataFrame({
            'account_id': np.repeat([1001, 1002, 1003, 1004], len(dates)),
            'balance': np.random.normal(10000, 1000, size=len(dates) * 4),
            'date': np.tile(dates, 4)
        })
        
        # Add some balance anomalies
        anomaly_indices = np.random.choice(len(self.mock_balance_data), size=5, replace=False)
        self.mock_balance_data.loc[anomaly_indices, 'balance'] *= 3

    def test_prepare_monitoring_data(self):
        """Test preparation of monitoring data"""
        # Mock data loading functions
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_balance_data = Mock(return_value=self.mock_balance_data)
        
        # Test data preparation
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        # Verify structure
        self.assertIsInstance(monitoring_data, pd.DataFrame)
        self.assertGreater(len(monitoring_data.columns), 0)
        
        # Verify feature categories
        self.assertTrue(any(col.startswith('transaction_') for col in monitoring_data.columns))
        self.assertTrue(any(col.startswith('balance_') for col in monitoring_data.columns))
        self.assertTrue(any(col.startswith('pattern_') for col in monitoring_data.columns))

    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Train the monitor first
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_balance_data = Mock(return_value=self.mock_balance_data)
        
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        self.monitor.train_monitor(monitoring_data)
        
        # Test anomaly detection
        detection_results = self.monitor.detect_anomalies(monitoring_data)
        
        # Verify structure
        self.assertIn('anomalies', detection_results)
        self.assertIn('detection_summary', detection_results)
        
        # Verify anomaly details
        anomalies = detection_results['anomalies']
        self.assertGreater(len(anomalies), 0)
        
        for anomaly in anomalies:
            self.assertIn('account_id', anomaly)
            self.assertIn('anomaly_score', anomaly)
            self.assertIn('anomaly_type', anomaly)
            self.assertIn('details', anomaly)

    def test_analyze_account_patterns(self):
        """Test account pattern analysis"""
        # Mock data loading functions
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_balance_data = Mock(return_value=self.mock_balance_data)
        
        # Test pattern analysis
        pattern_results = self.monitor.analyze_account_patterns(1001)  # Test for specific account
        
        # Verify structure
        self.assertIn('patterns', pattern_results)
        self.assertIn('anomalies', pattern_results)
        self.assertIn('metrics', pattern_results)
        
        # Verify pattern metrics
        patterns = pattern_results['patterns']
        self.assertIn('transaction_patterns', patterns)
        self.assertIn('balance_patterns', patterns)
        self.assertIn('timing_patterns', patterns)

    def test_calculate_anomaly_scores(self):
        """Test anomaly score calculation"""
        # Prepare test data
        test_data = pd.DataFrame({
            'transaction_volume': np.random.normal(1000, 100, size=100),
            'balance_change': np.random.normal(0, 1000, size=100),
            'pattern_complexity': np.random.uniform(0, 1, size=100)
        })
        
        # Add some known anomalies
        test_data.loc[0, 'transaction_volume'] = 10000  # Clear anomaly
        test_data.loc[1, 'balance_change'] = 10000  # Clear anomaly
        
        # Calculate scores
        anomaly_scores = self.monitor._calculate_anomaly_scores(
            test_data.values,  # Numpy array for isolation forest
            np.zeros(len(test_data))  # Mock predictions
        )
        
        # Verify scores
        self.assertEqual(len(anomaly_scores), len(test_data))
        self.assertTrue(all(0 <= score <= 1 for score in anomaly_scores['composite_score']))
        self.assertTrue(anomaly_scores.loc[0, 'composite_score'] > anomaly_scores['composite_score'].mean())

    def test_identify_anomalies(self):
        """Test anomaly identification and classification"""
        # Prepare test data with known anomalies
        test_features = pd.DataFrame({
            'transaction_volume': [1000] * 10,
            'balance_change': [100] * 10,
            'pattern_complexity': [0.5] * 10
        })
        
        # Add clear anomalies
        test_features.loc[0, 'transaction_volume'] = 10000
        test_features.loc[1, 'balance_change'] = 5000
        
        # Mock anomaly scores
        mock_scores = pd.DataFrame({
            'isolation_score': [0.9, 0.8] + [0.2] * 8,
            'composite_score': [0.9, 0.8] + [0.2] * 8
        })
        
        # Identify anomalies
        anomalies = self.monitor._identify_anomalies(
            test_features,
            np.array([-1, -1] + [1] * 8),  # First two are anomalies
            mock_scores
        )
        
        # Verify anomaly identification
        self.assertEqual(len(anomalies), 2)
        self.assertGreater(anomalies[0]['anomaly_score'], anomalies[1]['anomaly_score'])
        self.assertEqual(anomalies[0]['anomaly_type'], 'VOLUME_ANOMALY')

    def test_train_monitor(self):
        """Test monitor training"""
        # Prepare training data
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_balance_data = Mock(return_value=self.mock_balance_data)
        
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        # Train monitor
        training_results = self.monitor.train_monitor(monitoring_data)
        
        # Verify training results
        self.assertIn('model_metrics', training_results)
        self.assertIn('feature_importance', training_results)
        
        # Verify models were trained
        self.assertIsNotNone(self.monitor.isolation_forest)
        self.assertIsNotNone(self.monitor.envelope_model)
        self.assertIsNotNone(self.monitor.lof_model)

    def test_save_and_load_monitor(self):
        """Test saving and loading monitor state"""
        # Train monitor first
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_balance_data = Mock(return_value=self.mock_balance_data)
        
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        self.monitor.train_monitor(monitoring_data)
        
        # Save monitor state
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'monitor_state.joblib')
            self.monitor.save_monitor(save_path)
            
            # Create new monitor and load state
            new_monitor = AccountMonitor(Mock())
            new_monitor.load_monitor(save_path)
            
            # Verify loaded state
            self.assertEqual(
                self.monitor.feature_columns,
                new_monitor.feature_columns
            )
            self.assertEqual(
                len(self.monitor.model_metrics),
                len(new_monitor.model_metrics)
            )

    def test_analyze_detection_patterns(self):
        """Test analysis of detection patterns"""
        # Train monitor and generate some anomalies
        self.monitor._get_gl_data = Mock(return_value=self.mock_gl_data)
        self.monitor._get_balance_data = Mock(return_value=self.mock_balance_data)
        
        monitoring_data = self.monitor.prepare_monitoring_data(
            self.start_date,
            self.end_date
        )
        
        self.monitor.train_monitor(monitoring_data)
        detection_results = self.monitor.detect_anomalies(monitoring_data)
        
        # Analyze patterns
        pattern_analysis = self.monitor.analyze_detection_patterns(
            self.start_date,
            self.end_date
        )
        
        # Verify analysis structure
        self.assertIn('temporal_patterns', pattern_analysis)
        self.assertIn('factor_patterns', pattern_analysis)
        self.assertIn('risk_distribution', pattern_analysis)
        
        # Verify pattern details
        temporal = pattern_analysis['temporal_patterns']
        self.assertIn('daily_distribution', temporal)
        self.assertIn('hourly_distribution', temporal)
        self.assertIn('weekday_distribution', temporal)

if __name__ == '__main__':
    unittest.main()