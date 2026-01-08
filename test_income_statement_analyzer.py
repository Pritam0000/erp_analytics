# tests/analytics/financial_analysis/test_income_statement_analyzer.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.financial_analysis.income_statement_analyzer import IncomeStatementAnalyzer

class TestIncomeStatementAnalyzer(unittest.TestCase):
    """Test cases for IncomeStatementAnalyzer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.analyzer = IncomeStatementAnalyzer(Mock())
        # Mock database connection and logger
        self.analyzer.data_loader = Mock()
        self.analyzer.logger = Mock()
        
        # Set up test dates
        self.test_end_date = datetime(2024, 1, 31)
        self.test_start_date = datetime(2024, 1, 1)
        
        # Set up mock GL data
        self.mock_gl_coa = pd.DataFrame({
            'account_id': range(1, 11),
            'account_type': ['REVENUE'] * 3 + ['EXPENSE'] * 7,
            'account_category': [
                'OPERATING', 'OTHER', 'OPERATING',            # Revenue
                'COST_OF_SALES', 'OPERATING', 'OTHER',       # Expenses
                'COST_OF_SALES', 'OPERATING', 'OTHER', 'OPERATING'  # More Expenses
            ],
            'account_name': [f'Account_{i}' for i in range(1, 11)],
            'is_active': ['Y'] * 10
        })
        
        # Set up mock GL transactions
        self.mock_gl_journals = pd.DataFrame({
            'journal_id': range(1, 11),
            'journal_date': [
                self.test_start_date + timedelta(days=i) 
                for i in range(10)
            ],
            'status': ['POSTED'] * 10
        })
        
        # Set up mock journal lines with typical revenue/expense patterns
        self.mock_gl_lines = pd.DataFrame({
            'journal_id': range(1, 11),
            'account_id': range(1, 11),
            'debit_amount': [0, 0, 0, 2000, 1000, 500, 1500, 800, 300, 700],  # Expenses
            'credit_amount': [5000, 1000, 3000, 0, 0, 0, 0, 0, 0, 0]  # Revenue
        })
        
        # Configure mock data loader
        self.analyzer.data_loader.data = {
            'gl_coa': self.mock_gl_coa,
            'gl_journals': self.mock_gl_journals,
            'gl_lines': self.mock_gl_lines
        }

    def test_analyze_income_statement(self):
        """Test income statement analysis"""
        result = self.analyzer.analyze_income_statement(
            self.test_start_date,
            self.test_end_date
        )
        
        # Verify structure
        self.assertIn('period', result)
        self.assertIn('revenue', result)
        self.assertIn('expenses', result)
        self.assertIn('profit_metrics', result)
        self.assertIn('performance_metrics', result)
        self.assertIn('trend_analysis', result)
        
        # Verify totals
        self.assertEqual(result['revenue']['total'], 9000)  # Sum of revenue
        self.assertEqual(result['expenses']['total'], 6800)  # Sum of expenses
        
        # Verify profit calculations
        profit_metrics = result['profit_metrics']
        self.assertEqual(profit_metrics['gross_profit'], 5500)  # Revenue - COGS
        self.assertEqual(profit_metrics['operating_profit'], 3000)  # Gross - Operating
        self.assertEqual(profit_metrics['net_profit'], 2200)  # Operating - Other

    def test_calculate_revenue(self):
        """Test revenue calculation"""
        revenue = self.analyzer._calculate_revenue()
        
        # Verify revenue categories
        self.assertEqual(revenue['operating_revenue'], 8000)  # Accounts 1 and 3
        self.assertEqual(revenue['other_revenue'], 1000)     # Account 2
        self.assertEqual(revenue['total'], 9000)            # Total revenue

    def test_calculate_expenses(self):
        """Test expense calculation"""
        expenses = self.analyzer._calculate_expenses()
        
        # Verify expense categories
        self.assertEqual(expenses['cost_of_sales'], 3500)  # Accounts 4 and 7
        self.assertEqual(expenses['operating'], 2500)      # Accounts 5, 8, and 10
        self.assertEqual(expenses['other'], 800)          # Accounts 6 and 9
        self.assertEqual(expenses['total'], 6800)         # Total expenses

    def test_get_account_activity(self):
        """Test account activity calculation"""
        # Test revenue account (should have credit balance)
        revenue_activity = self.analyzer._get_account_activity(
            1,  # Revenue account ID
            self.mock_gl_lines,
            self.mock_gl_journals
        )
        self.assertEqual(revenue_activity, 5000)
        
        # Test expense account (should have debit balance)
        expense_activity = self.analyzer._get_account_activity(
            4,  # Expense account ID
            self.mock_gl_lines,
            self.mock_gl_journals
        )
        self.assertEqual(expense_activity, 2000)

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        revenue = {
            'total': 9000,
            'operating_revenue': 8000,
            'other_revenue': 1000
        }
        
        expenses = {
            'cost_of_sales': 3500,
            'operating': 2500,
            'other': 800,
            'total': 6800
        }
        
        gross_profit = revenue['total'] - expenses['cost_of_sales']
        operating_profit = gross_profit - expenses['operating']
        net_profit = operating_profit - expenses['other']
        
        metrics = self.analyzer._calculate_performance_metrics(
            revenue,
            expenses,
            gross_profit,
            operating_profit,
            net_profit
        )
        
        # Verify margin calculations
        self.assertAlmostEqual(metrics['gross_margin'], 61.11, places=2)
        self.assertAlmostEqual(metrics['operating_margin'], 33.33, places=2)
        self.assertAlmostEqual(metrics['net_margin'], 24.44, places=2)
        
        # Verify expense ratios
        self.assertAlmostEqual(metrics['cost_of_sales_ratio'], 38.89, places=2)
        self.assertAlmostEqual(metrics['operating_expense_ratio'], 27.78, places=2)
        self.assertAlmostEqual(metrics['other_expense_ratio'], 8.89, places=2)

    def test_analyze_trends(self):
        """Test trend analysis"""
        trends = self.analyzer._analyze_trends(months=3)
        
        # Verify trend structure
        self.assertIn('revenue', trends)
        self.assertIn('gross_profit', trends)
        self.assertIn('operating_profit', trends)
        self.assertIn('net_profit', trends)
        self.assertIn('margins', trends)
        
        # Verify trend data for each metric
        for metric in ['revenue', 'gross_profit', 'operating_profit', 'net_profit']:
            self.assertEqual(len(trends[metric]), 3)  # 3 months of data
            for point in trends[metric]:
                self.assertIn('date', point)
                self.assertIn('value', point)

    def test_get_income_statement_summary(self):
        """Test income statement summary generation"""
        # First analyze income statement
        self.analyzer.analyze_income_statement(
            self.test_start_date,
            self.test_end_date
        )
        
        # Get summary
        summary = self.analyzer.get_income_statement_summary()
        
        # Verify summary structure
        self.assertIn('period', summary)
        self.assertIn('revenue', summary)
        self.assertIn('expenses', summary)
        self.assertIn('profit', summary)
        self.assertIn('key_metrics', summary)
        
        # Verify key totals
        self.assertEqual(summary['revenue'], 9000)
        self.assertEqual(summary['expenses'], 6800)
        self.assertEqual(summary['profit']['net_profit'], 2200)

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Mock empty data
        self.analyzer.data_loader.data = {
            'gl_coa': pd.DataFrame(),
            'gl_journals': pd.DataFrame(),
            'gl_lines': pd.DataFrame()
        }
        
        result = self.analyzer.analyze_income_statement(
            self.test_start_date,
            self.test_end_date
        )
        
        # Verify zeros for all totals
        self.assertEqual(result['revenue']['total'], 0)
        self.assertEqual(result['expenses']['total'], 0)
        self.assertEqual(result['profit_metrics']['net_profit'], 0)

    def test_date_range_filtering(self):
        """Test proper date range filtering"""
        # Set up test with transactions outside date range
        extended_journals = self.mock_gl_journals.copy()
        extended_journals.loc[10] = {
            'journal_id': 11,
            'journal_date': self.test_end_date + timedelta(days=1),
            'status': 'POSTED'
        }
        
        extended_lines = self.mock_gl_lines.copy()
        extended_lines.loc[10] = {
            'journal_id': 11,
            'account_id': 1,
            'debit_amount': 0,
            'credit_amount': 1000
        }
        
        self.analyzer.data_loader.data.update({
            'gl_journals': extended_journals,
            'gl_lines': extended_lines
        })
        
        result = self.analyzer.analyze_income_statement(
            self.test_start_date,
            self.test_end_date
        )
        
        # Verify only transactions within date range are included
        self.assertEqual(result['revenue']['total'], 9000)  # Should not include extra 1000

    def test_invalid_date_handling(self):
        """Test handling of invalid date ranges"""
        # Test with end date before start date
        with self.assertRaises(ValueError):
            self.analyzer.analyze_income_statement(
                self.test_end_date,
                self.test_start_date
            )
        
        # Test with future dates
        future_date = datetime.now() + timedelta(days=365)
        result = self.analyzer.analyze_income_statement(
            future_date,
            future_date + timedelta(days=30)
        )
        
        # Verify zeros for all totals with future dates
        self.assertEqual(result['revenue']['total'], 0)
        self.assertEqual(result['expenses']['total'], 0)
        self.assertEqual(result['profit_metrics']['net_profit'], 0)

if __name__ == '__main__':
    unittest.main()