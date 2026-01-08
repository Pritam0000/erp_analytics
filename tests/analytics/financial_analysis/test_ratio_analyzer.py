# tests/analytics/financial_analysis/test_ratio_analyzer.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.financial_analysis.ratio_analyzer import RatioAnalyzer

class TestRatioAnalyzer(unittest.TestCase):
    """Test cases for RatioAnalyzer class"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create mock data loader
        data_loader = Mock()
        
        # Mock the analyze methods of other analyzer classes
        self.mock_bs_analysis = {
            'analysis_date': datetime(2024, 1, 1),
            'assets': {
                'details': {
                    'current_assets': 100000,
                    'fixed_assets': 200000,
                    'other_assets': 50000,
                    'inventory': 30000
                },
                'total': 350000
            },
            'liabilities': {
                'details': {
                    'current_liabilities': 50000,
                    'long_term_liabilities': 100000,
                    'other_liabilities': 25000
                },
                'total': 175000
            },
            'equity': {
                'details': {
                    'capital': 100000,
                    'retained_earnings': 60000,
                    'other_equity': 15000
                },
                'total': 175000
            }
        }
        
        self.mock_is_analysis = {
            'revenue': {
                'total': 500000,
                'operating_revenue': 450000,
                'other_revenue': 50000
            },
            'expenses': {
                'cost_of_sales': 300000,
                'operating': 100000,
                'other': 25000,
                'total': 425000
            },
            'profit_metrics': {
                'gross_profit': 200000,
                'operating_profit': 100000,
                'net_profit': 75000
            }
        }
        
        self.mock_cf_analysis = {
            'cash_flows': {
                'operating': {
                    'net_cash': 90000,
                    'collections_from_customers': 480000,
                    'payments_to_suppliers': -300000,
                    'operating_expenses': -90000
                },
                'investing': {
                    'net_cash': -50000,
                    'asset_purchases': -70000,
                    'asset_sales': 20000
                },
                'financing': {
                    'net_cash': -20000,
                    'debt_proceeds': 30000,
                    'debt_payments': -40000,
                    'dividends_paid': -10000
                },
                'net_cash_flow': 20000
            },
            'cash_positions': {
                'starting_cash': 40000,
                'ending_cash': 60000
            }
        }
        
        # Initialize analyzer with mock data
        self.analyzer = RatioAnalyzer(data_loader)
        self.analyzer.balance_sheet_analyzer = Mock()
        self.analyzer.income_statement_analyzer = Mock()
        self.analyzer.cash_flow_analyzer = Mock()
        
        # Configure mock return values
        self.analyzer.balance_sheet_analyzer.analyze_balance_sheet.return_value = self.mock_bs_analysis
        self.analyzer.income_statement_analyzer.analyze_income_statement.return_value = self.mock_is_analysis
        self.analyzer.cash_flow_analyzer.analyze_cash_flow.return_value = self.mock_cf_analysis
        
        # Set up test date
        self.test_date = datetime(2024, 1, 1)

    def test_calculate_ratios(self):
        """Test comprehensive ratio calculation"""
        result = self.analyzer.calculate_ratios(self.test_date)
        
        # Verify structure
        self.assertIn('analysis_date', result)
        self.assertIn('liquidity_ratios', result)
        self.assertIn('profitability_ratios', result)
        self.assertIn('efficiency_ratios', result)
        self.assertIn('solvency_ratios', result)
        self.assertIn('market_ratios', result)
        self.assertIn('cash_flow_ratios', result)
        self.assertIn('dupont_analysis', result)
        self.assertIn('ratio_trends', result)
        self.assertIn('benchmarks', result)

    def test_calculate_liquidity_ratios(self):
        """Test liquidity ratio calculations"""
        ratios = self.analyzer._calculate_liquidity_ratios(self.mock_bs_analysis)
        
        # Verify liquidity ratios
        self.assertEqual(ratios['current_ratio'], 2.0)  # 100000 / 50000
        self.assertEqual(ratios['quick_ratio'], 1.4)   # (100000 - 30000) / 50000
        self.assertEqual(ratios['working_capital'], 50000)  # 100000 - 50000
        self.assertAlmostEqual(ratios['working_capital_ratio'], 0.143, places=3)  # 50000 / 350000

    def test_calculate_profitability_ratios(self):
        """Test profitability ratio calculations"""
        ratios = self.analyzer._calculate_profitability_ratios(
            self.mock_bs_analysis,
            self.mock_is_analysis
        )
        
        # Verify profitability ratios
        self.assertEqual(ratios['gross_margin'], 40.0)  # (200000 / 500000) * 100
        self.assertEqual(ratios['operating_margin'], 20.0)  # (100000 / 500000) * 100
        self.assertEqual(ratios['net_profit_margin'], 15.0)  # (75000 / 500000) * 100
        self.assertAlmostEqual(ratios['return_on_assets'], 21.43, places=2)  # (75000 / 350000) * 100
        self.assertAlmostEqual(ratios['return_on_equity'], 42.86, places=2)  # (75000 / 175000) * 100

    def test_calculate_efficiency_ratios(self):
        """Test efficiency ratio calculations"""
        ratios = self.analyzer._calculate_efficiency_ratios(
            self.mock_bs_analysis,
            self.mock_is_analysis
        )
        
        # Verify efficiency ratios
        self.assertAlmostEqual(ratios['asset_turnover'], 1.43, places=2)  # 500000 / 350000
        self.assertEqual(ratios['inventory_turnover'], 10.0)  # 300000 / 30000
        self.assertEqual(ratios['days_inventory'], 36.5)  # 365 / 10

    def test_calculate_solvency_ratios(self):
        """Test solvency ratio calculations"""
        ratios = self.analyzer._calculate_solvency_ratios(self.mock_bs_analysis)
        
        # Verify solvency ratios
        self.assertEqual(ratios['debt_ratio'], 0.5)  # 175000 / 350000
        self.assertEqual(ratios['debt_to_equity'], 1.0)  # 175000 / 175000
        self.assertEqual(ratios['equity_ratio'], 0.5)  # 175000 / 350000

    def test_calculate_market_ratios(self):
        """Test market ratio calculations"""
        ratios = self.analyzer._calculate_market_ratios(
            self.mock_bs_analysis,
            self.mock_is_analysis
        )
        
        # Mock total shares method
        self.analyzer._get_total_shares = Mock(return_value=10000)
        
        # Calculate market ratios
        ratios = self.analyzer._calculate_market_ratios(
            self.mock_bs_analysis,
            self.mock_is_analysis
        )
        
        # Verify market ratios
        self.assertEqual(ratios['book_value_per_share'], 17.5)  # 175000 / 10000
        self.assertEqual(ratios['earnings_per_share'], 7.5)  # 75000 / 10000

    def test_calculate_cash_flow_ratios(self):
        """Test cash flow ratio calculations"""
        ratios = self.analyzer._calculate_cash_flow_ratios(
            self.mock_bs_analysis,
            self.mock_cf_analysis
        )
        
        # Verify cash flow ratios
        self.assertEqual(ratios['operating_cash_flow_ratio'], 1.8)  # 90000 / 50000
        self.assertAlmostEqual(ratios['cash_flow_coverage_ratio'], 0.514, places=3)  # 90000 / 175000
        self.assertAlmostEqual(ratios['cash_flow_to_assets'], 0.257, places=3)  # 90000 / 350000

    def test_perform_dupont_analysis(self):
        """Test DuPont analysis calculations"""
        analysis = self.analyzer._perform_dupont_analysis(
            self.mock_bs_analysis,
            self.mock_is_analysis
        )
        
        # Verify DuPont components
        self.assertEqual(analysis['net_profit_margin'], 15.0)  # (75000 / 500000) * 100
        self.assertAlmostEqual(analysis['asset_turnover'], 1.43, places=2)  # 500000 / 350000
        self.assertEqual(analysis['financial_leverage'], 2.0)  # 350000 / 175000
        
        # Verify ROE calculation
        expected_roe = 15.0 * 1.43 * 2.0 / 100  # Convert percentage to decimal
        self.assertAlmostEqual(analysis['roe_dupont'], expected_roe, places=2)

    def test_analyze_ratio_trends(self):
        """Test ratio trend analysis"""
        trends = self.analyzer._analyze_ratio_trends(months=3)
        
        # Verify trend structure
        self.assertIn('liquidity_trends', trends)
        self.assertIn('profitability_trends', trends)
        self.assertIn('efficiency_trends', trends)
        self.assertIn('solvency_trends', trends)
        
        # Verify trend data points
        for category in trends.values():
            self.assertEqual(len(category), 3)  # 3 months of data
            for point in category:
                self.assertIn('date', point)
                for key in ['current_ratio', 'gross_margin', 'asset_turnover', 'debt_ratio']:
                    if key in point:
                        self.assertIsInstance(point[key], (int, float))

    def test_get_ratio_summary(self):
        """Test ratio summary generation"""
        # First calculate ratios
        self.analyzer.calculate_ratios(self.test_date)
        
        # Get summary
        summary = self.analyzer.get_ratio_summary()
        
        # Verify summary structure
        self.assertIn('analysis_date', summary)
        self.assertIn('key_metrics', summary)
        self.assertIn('dupont_analysis', summary)
        self.assertIn('ratio_trends', summary)
        self.assertIn('performance_vs_benchmarks', summary)

    def test_compare_with_benchmarks(self):
        """Test benchmark comparison"""
        # First calculate ratios
        self.analyzer.calculate_ratios(self.test_date)
        
        # Compare with benchmarks
        comparison = self.analyzer._compare_with_benchmarks()
        
        # Verify comparison structure
        self.assertIn('liquidity', comparison)
        self.assertIn('profitability', comparison)
        self.assertIn('efficiency', comparison)
        self.assertIn('solvency', comparison)
        
        # Verify comparison metrics
        for category in comparison.values():
            for ratio in category.values():
                self.assertIn('value', ratio)
                self.assertIn('benchmark', ratio)
                self.assertIn('rating', ratio)

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Mock empty analysis results
        empty_bs = {
            'assets': {'total': 0, 'details': {}},
            'liabilities': {'total': 0, 'details': {}},
            'equity': {'total': 0, 'details': {}}
        }
        
        empty_is = {
            'revenue': {'total': 0},
            'expenses': {'total': 0},
            'profit_metrics': {'net_profit': 0}
        }
        
        empty_cf = {
            'cash_flows': {
                'operating': {'net_cash': 0},
                'investing': {'net_cash': 0},
                'financing': {'net_cash': 0}
            }
        }
        
        self.analyzer.balance_sheet_analyzer.analyze_balance_sheet.return_value = empty_bs
        self.analyzer.income_statement_analyzer.analyze_income_statement.return_value = empty_is
        self.analyzer.cash_flow_analyzer.analyze_cash_flow.return_value = empty_cf
        
        result = self.analyzer.calculate_ratios(self.test_date)
        
        # Verify zero/default values for ratios
        self.assertEqual(result['liquidity_ratios'].get('current_ratio', 0), 0)
        self.assertEqual(result['profitability_ratios'].get('return_on_equity', 0), 0)
        self.assertEqual(result['efficiency_ratios'].get('asset_turnover', 0), 0)

    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        # Test with negative values
        invalid_bs = {
            'assets': {'total': -1000, 'details': {'current_assets': -500}},
            'liabilities': {'total': -500, 'details': {'current_liabilities': -200}},
            'equity': {'total': -500, 'details': {}}
        }
        
        self.analyzer.balance_sheet_analyzer.analyze_balance_sheet.return_value = invalid_bs
        
        result = self.analyzer.calculate_ratios(self.test_date)
        
        # Verify ratios are handled appropriately with invalid data
        self.assertGreaterEqual(result['liquidity_ratios'].get('current_ratio', 0), 0)
        self.assertGreaterEqual(result['solvency_ratios'].get('debt_ratio', 0), 0)

if __name__ == '__main__':
    unittest.main()