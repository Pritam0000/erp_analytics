# tests/feature_engineering/test_temporal_features.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature_engineering.temporal_features.temporal_features import TemporalFeatureAnalyzer

class TestTemporalFeatureAnalyzer(unittest.TestCase):
    """Test cases for TemporalFeatureAnalyzer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.analyzer = TemporalFeatureAnalyzer()
        # Mock database connection and logger
        self.analyzer.db = Mock()
        self.analyzer.logger = Mock()
        
        # Set up test dates
        self.test_start_date = datetime(2024, 1, 1)
        self.test_end_date = datetime(2024, 3, 31)
        
        # Sample transaction timing data
        self.mock_transaction_data = pd.DataFrame({
            'hour_of_day': [9, 10, 11, 14, 15, 16],
            'day_of_week': [1, 2, 3, 1, 2, 3],
            'day_of_month': [1, 5, 10, 15, 20, 25],
            'transaction_count': [10, 15, 20, 25, 30, 40]  # Peak at hour 16
        })
        
        # Set up date range for seasonal data
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        seasonal_values = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 1000 + 2000
        
        # Create mock data matching SQL query output format
        self.mock_seasonal_data = pd.DataFrame({'date': dates, 'value': seasonal_values}).reset_index(drop=True)
        
        # Sample aging data
        self.mock_aging_data = pd.DataFrame({
            'aging_bucket': ['0-30', '31-60', '61-90', 'Over 90'],
            'item_count': [100, 50, 30, 20],
            'total_amount': [50000.0, 25000.0, 15000.0, 10000.0]
        })

    def test_analyze_transaction_timing(self):
        """Test transaction timing analysis"""
        # Set up mock return value
        self.analyzer.db.execute_query.return_value = self.mock_transaction_data

        # Test with default parameters
        result = self.analyzer.analyze_transaction_timing(
            start_date=self.test_start_date,
            end_date=self.test_end_date
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('timing_patterns', result)
        self.assertIn('period', result)

        # Verify timing patterns
        patterns = result['timing_patterns']
        self.assertIn('hourly', patterns)
        self.assertIn('daily', patterns)
        self.assertIn('monthly', patterns)

        # Verify peak calculations
        self.assertEqual(patterns['hourly']['peak_hour'], 16)  # Hour with most transactions
        self.assertEqual(patterns['daily']['peak_day'], 3)    # Day with most transactions

        # Verify that the expected query was made
        self.analyzer.db.execute_query.assert_called_once()
        call_args = self.analyzer.db.execute_query.call_args
        self.assertIn('start_date', call_args[1]['params'])
        self.assertIn('end_date', call_args[1]['params'])
        self.assertIn('transaction_type', call_args[1]['params'])

    def test_analyze_transaction_timing_empty_data(self):
        """Test transaction timing analysis with empty data"""
        # Set up mock to return empty DataFrame
        self.analyzer.db.execute_query.return_value = pd.DataFrame()

        result = self.analyzer.analyze_transaction_timing(
            start_date=self.test_start_date,
            end_date=self.test_end_date,
            transaction_type='PAYMENT'
        )

        # Verify handling of empty data
        self.assertEqual(result['status'], 'NO_DATA')
        self.assertIn('period', result)

    def test_analyze_transaction_timing_invalid_type(self):
        """Test transaction timing analysis with invalid transaction type"""
        # Set up mock to raise ValueError for invalid type
        self.analyzer.db.execute_query.side_effect = ValueError("Invalid transaction type")
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_transaction_timing(
                start_date=self.test_start_date,
                end_date=self.test_end_date,
                transaction_type='INVALID'
            )

    def test_analyze_seasonality(self):
        """Test seasonality analysis"""
        # Ensure mock data has the correct columns
        mock_data = self.mock_seasonal_data.copy()
        self.analyzer.db.execute_query.return_value = mock_data

        # Test with monthly frequency
        result = self.analyzer.analyze_seasonality(
            metric_type='PAYMENTS',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 31),
            frequency='ME'  # Using month end frequency
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('seasonality_metrics', result)
        self.assertIn('period', result)

        # Verify seasonality components
        metrics = result['seasonality_metrics']
        self.assertIn('seasonal_strength', metrics)
        self.assertIn('peak_period', metrics)
        self.assertIn('trough_period', metrics)
        self.assertIn('components', metrics)

        # Verify components structure
        components = metrics['components']
        self.assertIn('trend', components)
        self.assertIn('seasonal', components)
        self.assertIn('residual', components)

        # Verify that values are within expected ranges
        self.assertGreaterEqual(metrics['seasonal_strength'], 0)
        self.assertLessEqual(metrics['seasonal_strength'], 1)

    def test_analyze_seasonality_different_frequencies(self):
        """Test seasonality analysis with different frequencies"""
        # Ensure mock data has correct columns
        mock_data = self.mock_seasonal_data.copy()
        self.analyzer.db.execute_query.return_value = mock_data

        frequencies = ['D', 'W', 'ME']  # Using proper frequency codes
        for freq in frequencies:
            with self.subTest(frequency=freq):
                result = self.analyzer.analyze_seasonality(
                    metric_type='REVENUE',
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2024, 12, 31),
                    frequency=freq
                )
                self.assertEqual(result['period']['frequency'], freq)
                self.assertEqual(result['status'], 'ANALYZED')

    def test_analyze_seasonality_empty_data(self):
        """Test seasonality analysis with empty data"""
        # Set up mock to return empty DataFrame
        self.analyzer.db.execute_query.return_value = pd.DataFrame()

        result = self.analyzer.analyze_seasonality(
            metric_type='EXPENSES',
            start_date=self.test_start_date,
            end_date=self.test_end_date
        )

        # Verify handling of empty data
        self.assertEqual(result['status'], 'NO_DATA')
        self.assertIn('metric_type', result)

    def test_analyze_aging_metrics(self):
        """Test aging metrics analysis"""
        # Set up mock return value
        self.analyzer.db.execute_query.return_value = self.mock_aging_data

        # Test receivables aging
        result = self.analyzer.analyze_aging_metrics(
            metric_type='RECEIVABLES',
            as_of_date=self.test_end_date
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('aging_metrics', result)
        self.assertIn('as_of_date', result)

        # Verify aging metrics
        metrics = result['aging_metrics']
        self.assertIn('total_amount', metrics)
        self.assertIn('total_items', metrics)
        self.assertIn('aging_buckets', metrics)
        self.assertIn('risk_indicators', metrics)

        # Verify calculations
        self.assertEqual(metrics['total_items'], 200)  # Sum of item_count
        self.assertEqual(metrics['total_amount'], 100000.0)  # Sum of total_amount

        # Verify risk indicators
        risk = metrics['risk_indicators']
        self.assertIn('average_age', risk)
        self.assertIn('risk_level', risk)
        self.assertIn('old_bucket_concentration', risk)
        self.assertIn('recommended_actions', risk)

    def test_aging_risk_calculation(self):
        """Test aging risk calculation"""
        aging_stats = {
            '0-30': {'total_amount': 50000.0, 'amount_percentage': 50},
            '31-60': {'total_amount': 25000.0, 'amount_percentage': 25},
            '61-90': {'total_amount': 15000.0, 'amount_percentage': 15},
            'Over 90': {'total_amount': 10000.0, 'amount_percentage': 10}
        }

        risk_indicators = self.analyzer._calculate_aging_risk(aging_stats)

        # Verify risk indicators
        self.assertIn('average_age', risk_indicators)
        self.assertIn('risk_level', risk_indicators)
        self.assertEqual(risk_indicators['old_bucket_concentration'], 10.0)

        # Verify risk level determination
        self.assertIn(risk_indicators['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])

    def test_aging_recommendations(self):
        """Test aging recommendations generation"""
        # Test high risk scenario
        high_risk_recommendations = self.analyzer._get_aging_recommendations('HIGH', 25.0)
        self.assertGreater(len(high_risk_recommendations), 0)
        self.assertTrue(any('collection' in rec.lower() for rec in high_risk_recommendations))

        # Test medium risk scenario
        medium_risk_recommendations = self.analyzer._get_aging_recommendations('MEDIUM', 15.0)
        self.assertGreater(len(medium_risk_recommendations), 0)
        self.assertTrue(any('monitor' in rec.lower() for rec in medium_risk_recommendations))

        # Test low risk scenario
        low_risk_recommendations = self.analyzer._get_aging_recommendations('LOW', 5.0)
        self.assertEqual(len(low_risk_recommendations), 1)
        self.assertEqual(low_risk_recommendations[0], "Maintain current practices")

if __name__ == '__main__':
    unittest.main()