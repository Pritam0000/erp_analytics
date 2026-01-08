# tests/analytics/financial_analysis/test_cash_flow_analyzer.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.financial_analysis.cash_flow_analyzer import CashFlowAnalyzer

class TestCashFlowAnalyzer(unittest.TestCase):
    """Test cases for CashFlowAnalyzer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.analyzer = CashFlowAnalyzer(Mock())
        # Mock database connection and logger
        self.analyzer.data_loader = Mock()
        self.analyzer.logger = Mock()
        
        # Set up test dates
        self.test_end_date = datetime(2024, 1, 31)
        self.test_start_date = datetime(2024, 1, 1)
        
        # Set up mock GL data with a comprehensive chart of accounts
        self.mock_gl_coa = pd.DataFrame({
            'account_id': range(1, 21),
            'account_type': (
                ['ASSET'] * 5 +       # Cash and operating assets
                ['LIABILITY'] * 5 +   # Operating liabilities
                ['ASSET'] * 4 +       # Investing assets
                ['LIABILITY'] * 3 +   # Financing liabilities
                ['EQUITY'] * 3        # Equity accounts
            ),
            'account_category': (
                ['CASH', 'BANK', 'CASH_EQUIVALENT', 'RECEIVABLE', 'INVENTORY'] +  # Operating assets
                ['PAYABLE', 'ACCRUED', 'CURRENT', 'CURRENT', 'CURRENT'] +         # Operating liabilities
                ['FIXED', 'FIXED', 'INVESTMENT', 'INVESTMENT'] +                  # Investing assets
                ['LOAN', 'DEBT', 'BOND'] +                                       # Financing liabilities
                ['CAPITAL', 'DIVIDEND', 'RETAINED_EARNINGS']                     # Equity accounts
            ),
            'account_name': [f'Account_{i}' for i in range(1, 21)],
            'is_active': ['Y'] * 20
        })
        
        # Set up mock GL journals
        self.mock_gl_journals = pd.DataFrame({
            'journal_id': range(1, 21),
            'journal_date': [
                self.test_start_date + timedelta(days=i)
                for i in range(20)
            ],
            'status': ['POSTED'] * 20
        })
        
        # Set up mock journal lines with typical cash flow patterns
        self.mock_gl_lines = self._create_mock_gl_lines()
        
        # Set up mock AP data
        self.mock_ap_payments = pd.DataFrame({
            'payment_id': range(1, 6),
            'payment_date': [
                self.test_start_date + timedelta(days=i*2)
                for i in range(5)
            ],
            'amount': [1000, 1500, 2000, 1200, 1800],
            'status': ['POSTED'] * 5
        })
        
        # Configure mock data loader
        self.analyzer.data_loader.data = {
            'gl_coa': self.mock_gl_coa,
            'gl_journals': self.mock_gl_journals,
            'gl_lines': self.mock_gl_lines,
            'ap_payments': self.mock_ap_payments
        }

    def _create_mock_gl_lines(self):
        """Create detailed mock GL lines for different cash flow types"""
        # Operating activities
        operating_entries = [
            # Customer collections (cash increase)
            {'journal_id': 1, 'account_id': 1, 'debit_amount': 5000, 'credit_amount': 0},
            {'journal_id': 1, 'account_id': 4, 'debit_amount': 0, 'credit_amount': 5000},
            
            # Supplier payments (cash decrease)
            {'journal_id': 2, 'account_id': 6, 'debit_amount': 3000, 'credit_amount': 0},
            {'journal_id': 2, 'account_id': 1, 'debit_amount': 0, 'credit_amount': 3000},
            
            # Operating expenses (cash decrease)
            {'journal_id': 3, 'account_id': 7, 'debit_amount': 2000, 'credit_amount': 0},
            {'journal_id': 3, 'account_id': 1, 'debit_amount': 0, 'credit_amount': 2000}
        ]
        
        # Investing activities
        investing_entries = [
            # Asset purchase (cash decrease)
            {'journal_id': 4, 'account_id': 11, 'debit_amount': 10000, 'credit_amount': 0},
            {'journal_id': 4, 'account_id': 1, 'debit_amount': 0, 'credit_amount': 10000},
            
            # Investment sale (cash increase)
            {'journal_id': 5, 'account_id': 1, 'debit_amount': 8000, 'credit_amount': 0},
            {'journal_id': 5, 'account_id': 13, 'debit_amount': 0, 'credit_amount': 8000}
        ]
        
        # Financing activities
        financing_entries = [
            # Loan proceeds (cash increase)
            {'journal_id': 6, 'account_id': 1, 'debit_amount': 15000, 'credit_amount': 0},
            {'journal_id': 6, 'account_id': 15, 'debit_amount': 0, 'credit_amount': 15000},
            
            # Dividend payment (cash decrease)
            {'journal_id': 7, 'account_id': 19, 'debit_amount': 4000, 'credit_amount': 0},
            {'journal_id': 7, 'account_id': 1, 'debit_amount': 0, 'credit_amount': 4000}
        ]
        
        # Combine all entries
        all_entries = operating_entries + investing_entries + financing_entries
        return pd.DataFrame(all_entries)

    def test_analyze_cash_flow(self):
        """Test comprehensive cash flow analysis"""
        result = self.analyzer.analyze_cash_flow(
            self.test_start_date,
            self.test_end_date
        )
        
        # Verify structure
        self.assertIn('period', result)
        self.assertIn('cash_positions', result)
        self.assertIn('cash_flows', result)
        self.assertIn('metrics', result)
        self.assertIn('trend_analysis', result)
        
        # Verify cash flow components
        cash_flows = result['cash_flows']
        self.assertEqual(cash_flows['operating']['net_cash'], 0)  # Net operating cash flow
        self.assertEqual(cash_flows['investing']['net_cash'], -2000)  # Net investing cash flow
        self.assertEqual(cash_flows['financing']['net_cash'], 11000)  # Net financing cash flow
        self.assertEqual(cash_flows['net_cash_flow'], 9000)  # Total net cash flow

    def test_calculate_operating_cash_flow(self):
        """Test operating cash flow calculation"""
        operating_flows = self.analyzer._calculate_operating_cash_flow()
        
        # Verify operating cash flow components
        self.assertEqual(operating_flows['collections_from_customers'], 5000)
        self.assertEqual(operating_flows['payments_to_suppliers'], -3000)
        self.assertEqual(operating_flows['operating_expenses'], -2000)
        self.assertEqual(operating_flows['taxes_paid'], 0)
        self.assertEqual(operating_flows['other_operating'], 0)
        self.assertEqual(operating_flows['net_cash'], 0)

    def test_calculate_investing_cash_flow(self):
        """Test investing cash flow calculation"""
        investing_flows = self.analyzer._calculate_investing_cash_flow()
        
        # Verify investing cash flow components
        self.assertEqual(investing_flows['asset_purchases'], -10000)
        self.assertEqual(investing_flows['asset_sales'], 0)
        self.assertEqual(investing_flows['investment_purchases'], 0)
        self.assertEqual(investing_flows['investment_sales'], 8000)
        self.assertEqual(investing_flows['other_investing'], 0)
        self.assertEqual(investing_flows['net_cash'], -2000)

    def test_calculate_financing_cash_flow(self):
        """Test financing cash flow calculation"""
        financing_flows = self.analyzer._calculate_financing_cash_flow()
        
        # Verify financing cash flow components
        self.assertEqual(financing_flows['debt_proceeds'], 15000)
        self.assertEqual(financing_flows['debt_payments'], 0)
        self.assertEqual(financing_flows['equity_proceeds'], 0)
        self.assertEqual(financing_flows['dividends_paid'], -4000)
        self.assertEqual(financing_flows['other_financing'], 0)
        self.assertEqual(financing_flows['net_cash'], 11000)

    def test_get_cash_position(self):
        """Test cash position calculation"""
        # Test beginning cash position
        beginning_cash = self.analyzer._get_cash_position(self.test_start_date)
        self.assertEqual(beginning_cash, 0)  # Starting position
        
        # Test ending cash position
        ending_cash = self.analyzer._get_cash_position(self.test_end_date)
        self.assertEqual(ending_cash, 9000)  # Net change in cash

    def test_calculate_cash_flow_metrics(self):
        """Test cash flow metrics calculation"""
        operating_flows = self.analyzer._calculate_operating_cash_flow()
        investing_flows = self.analyzer._calculate_investing_cash_flow()
        financing_flows = self.analyzer._calculate_financing_cash_flow()
        starting_cash = self.analyzer._get_cash_position(self.test_start_date)
        
        metrics = self.analyzer._calculate_cash_flow_metrics(
            operating_flows,
            investing_flows,
            financing_flows,
            starting_cash
        )
        
        # Verify metrics calculations
        self.assertIn('operating_cash_ratio', metrics)
        self.assertIn('cash_sufficiency_ratio', metrics)
        self.assertIn('reinvestment_ratio', metrics)

    def test_analyze_trends(self):
        """Test cash flow trend analysis"""
        trends = self.analyzer._analyze_trends(months=3)
        
        # Verify trend structure
        self.assertIn('operating_cash_flow', trends)
        self.assertIn('investing_cash_flow', trends)
        self.assertIn('financing_cash_flow', trends)
        self.assertIn('net_cash_flow', trends)
        self.assertIn('cash_position', trends)
        
        # Verify trend data points
        for category in trends.values():
            self.assertEqual(len(category), 3)  # 3 months of data
            for point in category:
                self.assertIn('date', point)
                self.assertIn('value', point)

    def test_get_cash_flow_summary(self):
        """Test cash flow summary generation"""
        # First analyze cash flow
        self.analyzer.analyze_cash_flow(
            self.test_start_date,
            self.test_end_date
        )
        
        # Get summary
        summary = self.analyzer.get_cash_flow_summary()
        
        # Verify summary structure
        self.assertIn('period', summary)
        self.assertIn('cash_positions', summary)
        self.assertIn('cash_flows', summary)
        self.assertIn('key_metrics', summary)

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Mock empty data
        self.analyzer.data_loader.data = {
            'gl_coa': pd.DataFrame(),
            'gl_journals': pd.DataFrame(),
            'gl_lines': pd.DataFrame(),
            'ap_payments': pd.DataFrame()
        }
        
        result = self.analyzer.analyze_cash_flow(
            self.test_start_date,
            self.test_end_date
        )
        
        # Verify zeros for all cash flows
        self.assertEqual(result['cash_flows']['operating']['net_cash'], 0)
        self.assertEqual(result['cash_flows']['investing']['net_cash'], 0)
        self.assertEqual(result['cash_flows']['financing']['net_cash'], 0)
        self.assertEqual(result['cash_flows']['net_cash_flow'], 0)

    def test_date_range_filtering(self):
        """Test proper date range filtering"""
        # Add transaction outside date range
        extended_journals = self.mock_gl_journals.copy()
        extended_journals.loc[20] = {
            'journal_id': 21,
            'journal_date': self.test_end_date + timedelta(days=1),
            'status': 'POSTED'
        }
        
        extended_lines = self.mock_gl_lines.copy()
        extended_lines.loc[len(extended_lines)] = {
            'journal_id': 21,
            'account_id': 1,
            'debit_amount': 5000,
            'credit_amount': 0
        }
        
        self.analyzer.data_loader.data.update({
            'gl_journals': extended_journals,
            'gl_lines': extended_lines
        })
        
        result = self.analyzer.analyze_cash_flow(
            self.test_start_date,
            self.test_end_date
        )
        
        # Verify only transactions within date range are included
        self.assertEqual(result['cash_flows']['net_cash_flow'], 9000)

    def test_invalid_date_handling(self):
        """Test handling of invalid dates"""
        # Test with end date before start date
        with self.assertRaises(ValueError):
            self.analyzer.analyze_cash_flow(
                self.test_end_date,
                self.test_start_date
            )
            
        # Test with future dates
        future_date = datetime.now() + timedelta(days=365)
        result = self.analyzer.analyze_cash_flow(
            future_date,
            future_date + timedelta(days=30)
        )
        
        # Verify zeros for all cash flows with future dates
        self.assertEqual(result['cash_flows']['net_cash_flow'], 0)

if __name__ == '__main__':
    unittest.main()