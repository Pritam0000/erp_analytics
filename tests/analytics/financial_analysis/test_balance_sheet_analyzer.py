# tests/analytics/financial_analysis/test_balance_sheet_analyzer.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.financial_analysis.balance_sheet_analyzer import BalanceSheetAnalyzer

class TestBalanceSheetAnalyzer(unittest.TestCase):
    """Test cases for BalanceSheetAnalyzer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.analyzer = BalanceSheetAnalyzer(Mock())
        # Mock database connection and logger
        self.analyzer.data_loader = Mock()
        self.analyzer.logger = Mock()
        
        # Set up test date
        self.test_date = datetime(2024, 1, 1)
        
        # Set up mock GL data
        self.mock_gl_coa = pd.DataFrame({
            'account_id': range(1, 11),
            'account_type': ['ASSET'] * 4 + ['LIABILITY'] * 3 + ['EQUITY'] * 3,
            'account_category': [
                'CURRENT', 'FIXED', 'CURRENT', 'OTHER',  # Assets
                'CURRENT', 'LONG_TERM', 'OTHER',        # Liabilities
                'CAPITAL', 'RETAINED_EARNINGS', 'OTHER'  # Equity
            ],
            'account_name': [f'Account_{i}' for i in range(1, 11)],
            'is_active': ['Y'] * 10
        })
        
        # Set up mock GL transactions
        self.mock_gl_journals = pd.DataFrame({
            'journal_id': range(1, 6),
            'journal_date': [self.test_date - timedelta(days=i) for i in range(5)],
            'status': ['POSTED'] * 5
        })
        
        self.mock_gl_lines = pd.DataFrame({
            'journal_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'account_id': [1, 5, 2, 6, 3, 7, 4, 8, 9, 10],
            'debit_amount': [1000, 0, 2000, 0, 1500, 0, 500, 0, 0, 1000],
            'credit_amount': [0, 1000, 0, 2000, 0, 1500, 0, 500, 1000, 0]
        })
        
        # Configure mock data loader
        self.analyzer.data_loader.data = {
            'gl_coa': self.mock_gl_coa,
            'gl_journals': self.mock_gl_journals,
            'gl_lines': self.mock_gl_lines
        }

    def test_analyze_balance_sheet(self):
        """Test balance sheet analysis"""
        result = self.analyzer.analyze_balance_sheet(self.test_date)
        
        # Verify structure
        self.assertIn('analysis_date', result)
        self.assertIn('assets', result)
        self.assertIn('liabilities', result)
        self.assertIn('equity', result)
        self.assertIn('metrics', result)
        self.assertIn('trend_analysis', result)
        
        # Verify totals
        self.assertEqual(result['assets']['total'], 5000)  # Sum of asset balances
        self.assertEqual(result['liabilities']['total'], 3500)  # Sum of liability balances
        self.assertEqual(result['equity']['total'], 1500)  # Sum of equity balances
        
        # Verify balance sheet equation
        self.assertAlmostEqual(
            result['assets']['total'],
            result['liabilities']['total'] + result['equity']['total']
        )

    def test_calculate_assets(self):
        """Test asset calculation"""
        assets = self.analyzer._calculate_assets()
        
        # Verify individual asset categories
        self.assertEqual(assets['current_assets'], 2500)  # Accounts 1 and 3
        self.assertEqual(assets['fixed_assets'], 2000)   # Account 2
        self.assertEqual(assets['other_assets'], 500)    # Account 4
        
        # Verify total matches sum of categories
        total_assets = sum(assets.values())
        self.assertEqual(total_assets, 5000)

    def test_calculate_liabilities(self):
        """Test liability calculation"""
        liabilities = self.analyzer._calculate_liabilities()
        
        # Verify individual liability categories
        self.assertEqual(liabilities['current_liabilities'], 1000)   # Account 5
        self.assertEqual(liabilities['long_term_liabilities'], 2000) # Account 6
        self.assertEqual(liabilities['other_liabilities'], 500)      # Account 7
        
        # Verify total matches sum of categories
        total_liabilities = sum(liabilities.values())
        self.assertEqual(total_liabilities, 3500)

    def test_calculate_equity(self):
        """Test equity calculation"""
        equity = self.analyzer._calculate_equity()
        
        # Verify individual equity categories
        self.assertEqual(equity['capital'], 500)              # Account 8
        self.assertEqual(equity['retained_earnings'], 0)      # Account 9
        self.assertEqual(equity['other_equity'], 1000)        # Account 10
        
        # Verify total matches sum of categories
        total_equity = sum(equity.values())
        self.assertEqual(total_equity, 1500)

    def test_get_account_balance(self):
        """Test account balance calculation"""
        # Test asset account (should have debit balance)
        asset_balance = self.analyzer._get_account_balance(
            1,  # Asset account ID
            self.mock_gl_lines,
            self.mock_gl_journals
        )
        self.assertEqual(asset_balance, 1000)  # Debit balance
        
        # Test liability account (should have credit balance)
        liability_balance = self.analyzer._get_account_balance(
            5,  # Liability account ID
            self.mock_gl_lines,
            self.mock_gl_journals
        )
        self.assertEqual(liability_balance, -1000)  # Credit balance

    def test_calculate_balance_sheet_metrics(self):
        """Test balance sheet metrics calculation"""
        assets = {
            'current_assets': 2500,
            'fixed_assets': 2000,
            'other_assets': 500
        }
        
        liabilities = {
            'current_liabilities': 1000,
            'long_term_liabilities': 2000,
            'other_liabilities': 500
        }
        
        equity = {
            'capital': 500,
            'retained_earnings': 0,
            'other_equity': 1000
        }
        
        metrics = self.analyzer._calculate_balance_sheet_metrics(
            assets, liabilities, equity
        )
        
        # Verify key metrics
        self.assertEqual(metrics['working_capital'], 1500)  # 2500 - 1000
        self.assertEqual(metrics['current_ratio'], 2.5)     # 2500 / 1000
        self.assertEqual(metrics['debt_to_equity'], 2.33)   # 3500 / 1500
        self.assertEqual(metrics['fixed_asset_ratio'], 0.4)  # 2000 / 5000

    def test_analyze_trends(self):
        """Test trend analysis"""
        trends = self.analyzer._analyze_trends(months=3)
        
        # Verify trend structure
        self.assertIn('total_assets', trends)
        self.assertIn('total_liabilities', trends)
        self.assertIn('total_equity', trends)
        self.assertIn('working_capital', trends)
        self.assertIn('current_ratio', trends)
        
        # Verify trend data
        for metric in trends.values():
            self.assertEqual(len(metric), 3)  # 3 months of data
            for point in metric:
                self.assertIn('date', point)
                self.assertIn('value', point)

    def test_get_balance_sheet_summary(self):
        """Test balance sheet summary generation"""
        # First analyze balance sheet
        self.analyzer.analyze_balance_sheet(self.test_date)
        
        # Get summary
        summary = self.analyzer.get_balance_sheet_summary()
        
        # Verify summary structure
        self.assertIn('date', summary)
        self.assertIn('assets', summary)
        self.assertIn('liabilities', summary)
        self.assertIn('equity', summary)
        self.assertIn('key_metrics', summary)
        
        # Verify totals in summary
        self.assertEqual(summary['assets']['total'], 5000)
        self.assertEqual(summary['liabilities']['total'], 3500)
        self.assertEqual(summary['equity']['total'], 1500)

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Mock empty data
        self.analyzer.data_loader.data = {
            'gl_coa': pd.DataFrame(),
            'gl_journals': pd.DataFrame(),
            'gl_lines': pd.DataFrame()
        }
        
        result = self.analyzer.analyze_balance_sheet(self.test_date)
        
        # Verify zeros for all totals
        self.assertEqual(result['assets']['total'], 0)
        self.assertEqual(result['liabilities']['total'], 0)
        self.assertEqual(result['equity']['total'], 0)

    def test_invalid_date_handling(self):
        """Test handling of invalid analysis date"""
        future_date = datetime.now() + timedelta(days=365)
        result = self.analyzer.analyze_balance_sheet(future_date)
        
        # Verify zeros for all totals with future date
        self.assertEqual(result['assets']['total'], 0)
        self.assertEqual(result['liabilities']['total'], 0)
        self.assertEqual(result['equity']['total'], 0)

if __name__ == '__main__':
    unittest.main()