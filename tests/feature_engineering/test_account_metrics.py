# tests/feature_engineering/test_account_metrics.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature_engineering.financial_metrics.account_metrics import AccountMetricsCalculator

class TestAccountMetricsCalculator(unittest.TestCase):
    """Test cases for AccountMetricsCalculator class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.calculator = AccountMetricsCalculator()
        # Mock database connection
        self.calculator.db = Mock()
        
        # Set up common test data
        self.test_account_id = 2020
        self.test_date = datetime(2024, 1, 1)
        
        # Set up mock activity data for reuse
        self.mock_activity_data = pd.DataFrame({
            'transaction_count': [50],
            'journal_count': [20],
            'last_activity_date': [self.test_date],
            'debit_count': [30],
            'credit_count': [20],
            'max_debit': [1000.0],
            'max_credit': [800.0]
        })
        
        # Set up mock trend data for reuse
        # Using 'ME' (month end) instead of deprecated 'M'
        self.mock_trend_data = pd.DataFrame({
            'month_start': pd.date_range(start='2023-01-01', periods=12, freq='ME'),
            'monthly_debits': [1000.0] * 12,
            'monthly_credits': [800.0] * 12,
            'transaction_count': [10] * 12
        })

    def test_calculate_account_balance(self):
        """Test account balance calculation"""
        # Mock database response
        mock_balance_data = pd.DataFrame({
            'total_debits': [1000.0],
            'total_credits': [600.0]
        })
        self.calculator.db.execute_query.return_value = mock_balance_data

        # Calculate balance
        result = self.calculator.calculate_account_balance(
            self.test_account_id, 
            self.test_date
        )

        # Verify results
        self.assertEqual(result['balance'], 400.0)
        self.assertEqual(result['total_debits'], 1000.0)
        self.assertEqual(result['total_credits'], 600.0)

        # Verify query parameters
        self.calculator.db.execute_query.assert_called_once()
        call_args = self.calculator.db.execute_query.call_args
        self.assertIn('account_id', call_args[1]['params'])
        self.assertIn('as_of_date', call_args[1]['params'])

    def test_empty_account_balance(self):
        """Test balance calculation for account with no transactions"""
        # Mock empty database response
        self.calculator.db.execute_query.return_value = pd.DataFrame()

        # Calculate balance
        result = self.calculator.calculate_account_balance(
            self.test_account_id, 
            self.test_date
        )

        # Verify results show zero balances
        self.assertEqual(result['balance'], 0.0)
        self.assertEqual(result['total_debits'], 0.0)
        self.assertEqual(result['total_credits'], 0.0)

    def test_get_account_activity_metrics(self):
        """Test account activity metrics calculation"""
        # Set mock return value
        self.calculator.db.execute_query.return_value = self.mock_activity_data

        # Calculate activity metrics
        start_date = self.test_date - timedelta(days=90)
        result = self.calculator.get_account_activity_metrics(
            self.test_account_id,
            start_date,
            self.test_date
        )

        # Verify results
        self.assertEqual(result['transaction_count'], 50)
        self.assertEqual(result['journal_count'], 20)
        self.assertEqual(result['debit_count'], 30)
        self.assertEqual(result['credit_count'], 20)
        self.assertEqual(result['max_debit'], 1000.0)
        self.assertEqual(result['max_credit'], 800.0)
        self.assertIn('activity_level', result)

    def test_calculate_balance_trends(self):
        """Test balance trends calculation"""
        # Set mock return value
        self.calculator.db.execute_query.return_value = self.mock_trend_data

        # Calculate trends
        result = self.calculator.calculate_balance_trends(
            self.test_account_id,
            months=12
        )

        # Verify results
        self.assertEqual(len(result), 12)  # 12 months of data
        self.assertTrue('net_change' in result.columns)
        self.assertTrue('running_balance' in result.columns)
        self.assertTrue('balance_change_pct' in result.columns)

        # Verify calculations
        self.assertTrue((result['net_change'] == 200.0).all())  # 1000 - 800 = 200
        
        # Check if running balance is consistently increasing
        running_balance_values = result['running_balance'].values
        self.assertTrue(all(running_balance_values[i] <= running_balance_values[i + 1] 
                          for i in range(len(running_balance_values) - 1)))

    def test_get_account_summary(self):
        """Test comprehensive account summary generation"""
        # Mock account data
        mock_account_data = pd.DataFrame({
            'account_number': ['AC1234'],
            'account_name': ['Test Account'],
            'account_type': ['ASSET'],
            'account_category': ['CURRENT'],
            'is_active': ['Y']
        })
        
        # Set up multiple mock returns
        self.calculator.db.execute_query.side_effect = [
            mock_account_data,  # For account details
            pd.DataFrame({'total_debits': [1000.0], 'total_credits': [600.0]}),  # For balance
            self.mock_activity_data,  # For activity metrics
            self.mock_trend_data  # For trends
        ]

        # Get account summary
        result = self.calculator.get_account_summary(self.test_account_id)

        # Verify structure and content
        self.assertIn('account_info', result)
        self.assertIn('current_balance', result)
        self.assertIn('recent_activity', result)
        self.assertIn('trends', result)

        # Verify account info
        self.assertEqual(result['account_info']['account_number'], 'AC1234')
        self.assertEqual(result['account_info']['account_type'], 'ASSET')

        # Verify balance calculation
        self.assertEqual(result['current_balance']['balance'], 400.0)

        # Verify trends analysis
        self.assertIn('average_monthly_volume', result['trends'])
        self.assertIn('balance_volatility', result['trends'])
        self.assertIn('trend_direction', result['trends'])

    def test_determine_activity_level(self):
        """Test activity level determination"""
        test_cases = [
            (0, 'INACTIVE'),
            (5, 'LOW'),
            (30, 'MEDIUM'),
            (75, 'HIGH'),
            (150, 'VERY_HIGH')
        ]

        for transactions, expected_level in test_cases:
            with self.subTest(transactions=transactions):
                result = self.calculator._determine_activity_level(transactions)
                self.assertEqual(result, expected_level)

if __name__ == '__main__':
    unittest.main()