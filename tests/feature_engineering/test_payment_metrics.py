# tests/feature_engineering/test_journal_metrics.py

import sys
import os
import unittest
from datetime import datetime
import pandas as pd
from unittest.mock import Mock

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature_engineering.financial_metrics.journal_metrics import (
    calculate_journal_entry_count,
    calculate_journal_line_count,
    calculate_journal_balance,
    get_unbalanced_journals
)

class TestJournalMetrics(unittest.TestCase):
    """Test cases for journal metrics functions"""

    def setUp(self):
        """Set up test data before each test"""
        self.test_period = '2023-01'
        self.test_journal_id = 1

        self.mock_journals_data = pd.DataFrame({
            'journal_id': [1, 2, 3, 4, 5],
            'period_name': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-03']
        })

        self.mock_journal_lines_data = pd.DataFrame({
            'line_id': [1, 2, 3, 4, 5, 6],
            'journal_id': [1, 1, 2, 2, 3, 3],
            'debit_amount': [100, 200, 150, 150, 200, 100],
            'credit_amount': [0, 300, 0, 150, 0, 250]
        })

    def test_calculate_journal_entry_count(self):
        """Test journal entry count calculation"""
        result = calculate_journal_entry_count(self.mock_journals_data, self.test_period)
        self.assertEqual(result, 2)

    def test_calculate_journal_line_count(self):
        """Test journal line count calculation"""
        result = calculate_journal_line_count(self.mock_journal_lines_data, self.test_journal_id)
        self.assertEqual(result, 2)

    def test_calculate_journal_balance(self):
        """Test journal balance calculation"""
        result = calculate_journal_balance(self.mock_journal_lines_data, self.test_journal_id)
        self.assertEqual(result, 300)

    def test_get_unbalanced_journals(self):
        """Test unbalanced journal identification"""
        result = get_unbalanced_journals(self.mock_journal_lines_data)
        self.assertListEqual(result, [1, 3])

    def test_get_unbalanced_journals_with_tolerance(self):
        """Test unbalanced journal identification with tolerance"""
        result = get_unbalanced_journals(self.mock_journal_lines_data, tolerance=50)
        self.assertListEqual(result, [3])

if __name__ == '__main__':
    unittest.main()