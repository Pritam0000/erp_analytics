# tests/feature_engineering/test_journal_metrics.py

import sys
import os
import unittest
from datetime import datetime
import pandas as pd
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature_engineering.financial_metrics.journal_metrics import JournalMetricsCalculator

class TestJournalMetricsCalculator(unittest.TestCase):
    """Test cases for JournalMetricsCalculator class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.calculator = JournalMetricsCalculator()
        # Mock database connection
        self.calculator.db = Mock()
        self.calculator.logger = Mock()
        
        # Set up test data
        self.test_journal_id = 1234
        self.test_period = '2024-01'
        
        # Sample journal data with realistic values
        self.mock_journal_data = pd.DataFrame({
            'journal_id': [self.test_journal_id],
            'journal_number': ['JE001'],
            'journal_date': [datetime(2024, 1, 1)],
            'status': ['POSTED'],
            'period_name': [self.test_period],
            'unique_accounts': [3],
            'line_count': [6],
            'total_debits': [1000.0],
            'total_credits': [1000.0],
            'max_debit': [600.0],
            'max_credit': [400.0]
        })
        
        # Sample pattern analysis data
        self.mock_pattern_data = pd.DataFrame({
            'total_journals': [50],
            'avg_lines_per_journal': [4.5],
            'avg_accounts_per_journal': [2.8],
            'balanced_journals': [48],
            'avg_journal_amount': [5000.0],
            'max_journal_amount': [15000.0]
        })
        
        # Sample unbalanced journals data
        self.mock_unbalanced_data = pd.DataFrame({
            'journal_id': [1001, 1002],
            'journal_number': ['JE002', 'JE003'],
            'journal_date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'status': ['POSTED', 'POSTED'],
            'total_debits': [2000.0, 1000.0],
            'total_credits': [1800.0, 950.0],
            'balance_difference': [200.0, 50.0]
        })

    def test_calculate_journal_metrics(self):
        """Test journal metrics calculation"""
        # Set up mock return value
        self.calculator.db.execute_query.return_value = self.mock_journal_data

        # Calculate metrics
        result = self.calculator.calculate_journal_metrics(self.test_journal_id)

        # Verify structure
        self.assertIn('journal_info', result)
        self.assertIn('balance_metrics', result)
        self.assertIn('composition_metrics', result)
        self.assertIn('risk_indicators', result)

        # Verify balance calculations
        self.assertEqual(result['balance_metrics']['total_debits'], 1000.0)
        self.assertEqual(result['balance_metrics']['total_credits'], 1000.0)
        self.assertTrue(result['balance_metrics']['is_balanced'])

        # Verify composition metrics
        self.assertEqual(result['composition_metrics']['line_count'], 6)
        self.assertEqual(result['composition_metrics']['unique_accounts'], 3)

        # Verify DB query
        self.calculator.db.execute_query.assert_called_once()
        call_args = self.calculator.db.execute_query.call_args
        self.assertIn('journal_id', call_args[1]['params'])

    def test_analyze_journal_patterns(self):
        """Test journal pattern analysis"""
        # Set up mock return value
        self.calculator.db.execute_query.return_value = self.mock_pattern_data

        # Analyze patterns
        result = self.calculator.analyze_journal_patterns(self.test_period)

        # Verify structure
        self.assertIn('volume_metrics', result)
        self.assertIn('composition_metrics', result)
        self.assertIn('value_metrics', result)

        # Verify calculations
        volume_metrics = result['volume_metrics']
        self.assertEqual(volume_metrics['total_journals'], 50)
        self.assertEqual(volume_metrics['balanced_journals'], 48)
        self.assertAlmostEqual(volume_metrics['balance_rate'], 96.0)

        # Verify composition metrics
        comp_metrics = result['composition_metrics']
        self.assertAlmostEqual(comp_metrics['avg_lines_per_journal'], 4.5)
        self.assertAlmostEqual(comp_metrics['avg_accounts_per_journal'], 2.8)

    def test_get_unbalanced_journals(self):
        """Test unbalanced journal identification"""
        # Set up mock return value
        self.calculator.db.execute_query.return_value = self.mock_unbalanced_data

        # Get unbalanced journals
        result = self.calculator.get_unbalanced_journals(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            tolerance=0.01
        )

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0]['balance_difference'], result[1]['balance_difference'])

        # Verify query parameters
        call_args = self.calculator.db.execute_query.call_args
        params = call_args[1]['params']
        self.assertIn('tolerance', params)
        self.assertIn('start_date', params)
        self.assertIn('end_date', params)

    def test_complexity_score_calculation(self):
        """Test journal complexity score calculation"""
        test_cases = [
            (2, 1, 'SIMPLE'),     # Small, simple journal
            (4, 3, 'MODERATE'),   # Moderate complexity
            (8, 6, 'COMPLEX'),    # Complex journal
            (15, 10, 'VERY_COMPLEX')  # Very complex journal
        ]

        for line_count, unique_accounts, expected_score in test_cases:
            with self.subTest(line_count=line_count, unique_accounts=unique_accounts):
                score = self.calculator._calculate_complexity_score(
                    line_count, unique_accounts
                )
                self.assertEqual(score, expected_score)

    def test_risk_assessment(self):
        """Test risk assessment calculations"""
        # Test size risk assessment
        self.assertEqual(self.calculator._assess_size_risk(500.0), 'LOW')
        self.assertEqual(self.calculator._assess_size_risk(5000.0), 'MEDIUM')
        self.assertEqual(self.calculator._assess_size_risk(15000.0), 'HIGH')

        # Test complexity risk assessment
        self.assertEqual(self.calculator._assess_complexity_risk(2, 1), 'LOW')
        self.assertEqual(self.calculator._assess_complexity_risk(5, 4), 'MEDIUM')
        self.assertEqual(self.calculator._assess_complexity_risk(10, 8), 'HIGH')

    def test_empty_data_handling(self):
        """Test handling of empty or missing data"""
        # Test with empty journal data
        self.calculator.db.execute_query.return_value = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_journal_metrics(999999)

        # Test pattern analysis with no data
        result = self.calculator.analyze_journal_patterns('2099-12')
        self.assertEqual(result['status'], 'NO_DATA')

        # Test unbalanced journals with no data
        result = self.calculator.get_unbalanced_journals()
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()