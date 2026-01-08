# tests/feature_engineering/test_asset_metrics.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature_engineering.financial_metrics.asset_metrics import AssetMetricsCalculator

class TestAssetMetricsCalculator(unittest.TestCase):
    """Test cases for AssetMetricsCalculator class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.calculator = AssetMetricsCalculator()
        # Mock database connection and logger
        self.calculator.db = Mock()
        self.calculator.logger = Mock()
        
        # Set up test data
        self.test_asset_id = 1001
        self.test_category_id = 101
        self.test_date = datetime(2024, 1, 1)
        
        # Sample asset data with realistic values and proper types
        self.mock_asset_data = pd.DataFrame({
            'asset_id': [self.test_asset_id],
            'asset_number': ['AST1001'],
            'asset_name': ['Test Asset'],
            'category_id': [self.test_category_id],
            'category_name': ['Test Category'],
            'acquisition_date': [datetime(2023, 1, 1)],
            'acquisition_cost': [10000.0],
            'salvage_value': [1000.0],
            'life_years': [5],
            'status': ['IN_SERVICE'],
            'depreciation_method': ['STRAIGHT_LINE']
        })
        
        # Add book_value and accumulated_depreciation with potential NULL values
        self.mock_asset_data_with_depn = self.mock_asset_data.copy()
        self.mock_asset_data_with_depn['book_value'] = [8000.0]
        self.mock_asset_data_with_depn['accumulated_depreciation'] = [2000.0]
        
        # Sample depreciation data with monthly entries
        self.mock_depreciation_data = pd.DataFrame({
            'period_date': pd.date_range(start='2023-01-01', periods=12, freq='ME'),
            'depreciation_amount': [150.0] * 12,
            'accumulated_depreciation': [i * 150.0 for i in range(1, 13)],
            'book_value': [10000.0 - (i * 150.0) for i in range(1, 13)]
        })
        
        # Sample category data with realistic performance metrics
        self.mock_category_data = pd.DataFrame({
            'category_id': [self.test_category_id],
            'category_name': ['Test Category'],
            'depreciation_method': ['STRAIGHT_LINE'],
            'asset_count': [10],
            'total_acquisition_cost': [100000.0],
            'total_salvage_value': [10000.0],
            'avg_life_years': [5.0],
            'active_assets': [8],
            'retired_assets': [1],
            'disposed_assets': [1],
            'utilization_rate': [80.0]
        })

    def test_calculate_asset_depreciation(self):
        """Test asset depreciation calculation"""
        # Test with complete data
        self.calculator.db.execute_query.return_value = self.mock_asset_data_with_depn

        result = self.calculator.calculate_asset_depreciation(
            self.test_asset_id,
            self.test_date
        )

        # Verify structure and values
        self.assertIn('acquisition_cost', result)
        self.assertIn('current_book_value', result)
        self.assertIn('accumulated_depreciation', result)
        self.assertIn('annual_depreciation_rate', result)
        self.assertIn('remaining_life_years', result)
        self.assertIn('depreciation_percentage', result)

        # Verify calculations for straight-line method
        expected_annual_rate = ((10000.0 - 1000.0) / 5) / 10000.0 * 100
        self.assertAlmostEqual(result['annual_depreciation_rate'], expected_annual_rate, places=2)

        # Test with NULL depreciation values
        mock_data_nulls = self.mock_asset_data.copy()
        mock_data_nulls['book_value'] = [None]
        mock_data_nulls['accumulated_depreciation'] = [None]
        
        self.calculator.db.execute_query.return_value = mock_data_nulls
        
        result_nulls = self.calculator.calculate_asset_depreciation(
            self.test_asset_id,
            self.test_date
        )
        
        # Verify expected calculated values are used when actuals are NULL
        self.assertIsNotNone(result_nulls['current_book_value'])
        self.assertIsNotNone(result_nulls['accumulated_depreciation'])

    def test_calculate_different_depreciation_methods(self):
        """Test different depreciation calculation methods"""
        test_data = {
            'acquisition_cost': 10000.0,
            'salvage_value': 1000.0,
            'life_years': 5
        }

        # Test straight-line
        annual_sl = self.calculator._calculate_annual_depreciation(
            **test_data,
            depreciation_method='STRAIGHT_LINE'
        )
        expected_sl = (test_data['acquisition_cost'] - test_data['salvage_value']) / test_data['life_years']
        self.assertAlmostEqual(annual_sl, expected_sl, places=2)

        # Test declining balance
        annual_db = self.calculator._calculate_annual_depreciation(
            **test_data,
            depreciation_method='DECLINING_BALANCE'
        )
        expected_db = test_data['acquisition_cost'] * (2.0 / test_data['life_years'])
        self.assertAlmostEqual(annual_db, expected_db, places=2)

        # Test sum of years
        annual_soy = self.calculator._calculate_annual_depreciation(
            **test_data,
            depreciation_method='SUM_OF_YEARS'
        )
        sum_of_years = (test_data['life_years'] * (test_data['life_years'] + 1)) / 2
        expected_soy = (test_data['acquisition_cost'] - test_data['salvage_value']) * (test_data['life_years'] / sum_of_years)
        self.assertAlmostEqual(annual_soy, expected_soy, places=2)

    def test_analyze_category_performance(self):
        """Test category performance analysis"""
        self.calculator.db.execute_query.return_value = self.mock_category_data

        # Test with specific category
        result = self.calculator.analyze_category_performance(
            category_id=self.test_category_id
        )

        # Verify structure and calculations
        self.assertEqual(len(result), 1)
        self.assertTrue('utilization_rate' in result.columns)
        self.assertTrue('value_retention' in result.columns)
        self.assertEqual(result.iloc[0]['asset_count'], 10)
        self.assertEqual(result.iloc[0]['utilization_rate'], 80.0)

        # Verify value retention calculation
        expected_retention = ((100000.0 - 10000.0) / 100000.0) * 100
        self.assertAlmostEqual(result.iloc[0]['value_retention'], expected_retention, places=2)

    def test_get_asset_lifecycle_metrics(self):
        """Test asset lifecycle metrics calculation"""
        # Create asset info mock data
        mock_asset_info = pd.DataFrame({
            'asset_id': [self.test_asset_id],
            'asset_number': ['AST1001'],
            'asset_name': ['Test Asset'],
            'category_id': [self.test_category_id],
            'category_name': ['Test Category'],
            'acquisition_date': [datetime(2023, 1, 1)],
            'acquisition_cost': [10000.0],
            'salvage_value': [1000.0],
            'life_years': [5],
            'status': ['IN_SERVICE'],
            'depreciation_method': ['STRAIGHT_LINE'],
            'book_value': [8000.0],
            'accumulated_depreciation': [2000.0]
        })

        # Set up the sequence of mock returns for multiple queries
        self.calculator.db.execute_query.side_effect = [
            mock_asset_info,              # For get_asset_lifecycle_metrics initial query
            mock_asset_info,              # For calculate_asset_depreciation query
            self.mock_depreciation_data   # For depreciation history query
        ]

        result = self.calculator.get_asset_lifecycle_metrics(
            self.test_asset_id
        )

        # Verify structure
        self.assertIn('asset_info', result)
        self.assertIn('lifecycle_metrics', result)
        self.assertIn('financial_metrics', result)
        self.assertIn('depreciation_history', result)
        self.assertIn('performance_indicators', result)

        # Verify lifecycle calculations
        lifecycle = result['lifecycle_metrics']
        self.assertTrue(0 <= lifecycle['percent_through_lifecycle'] <= 100)
        self.assertGreater(lifecycle['age_years'], 0)
        self.assertIn(lifecycle['lifecycle_stage'], 
                     ['EARLY_LIFE', 'MID_LIFE', 'MATURE', 'LATE_LIFE', 'END_OF_LIFE'])

        # Verify financial calculations
        financial = result['financial_metrics']
        self.assertGreater(financial['total_depreciation'], 0)
        self.assertTrue(0 <= financial['value_retention'] <= 100)

        # Verify that the expected queries were made
        self.assertEqual(self.calculator.db.execute_query.call_count, 3)
        calls = self.calculator.db.execute_query.call_args_list
        self.assertIn('asset_id', calls[0][1]['params'])  # First query should have asset_id param

    def test_determine_lifecycle_stage(self):
        """Test lifecycle stage determination with realistic scenarios"""
        test_cases = [
            (0.8, 5, 'IN_SERVICE', 'EARLY_LIFE'),    # 16% through lifecycle
            (1.8, 5, 'IN_SERVICE', 'MID_LIFE'),      # 36% through lifecycle
            (2.8, 5, 'IN_SERVICE', 'MATURE'),        # 56% through lifecycle
            (3.8, 5, 'IN_SERVICE', 'LATE_LIFE'),     # 76% through lifecycle
            (4.5, 5, 'IN_SERVICE', 'END_OF_LIFE'),   # 90% through lifecycle
            (2.0, 5, 'RETIRED', 'RETIRED'),
            (3.0, 5, 'DISPOSED', 'DISPOSED')
        ]

        for years_held, life_years, status, expected_stage in test_cases:
            with self.subTest(years_held=years_held, status=status):
                result = self.calculator._determine_lifecycle_stage(
                    years_held, life_years, status
                )
                self.assertEqual(result, expected_stage, 
                    f"Expected {expected_stage} for {years_held} years (lifecycle percentage: {(years_held/life_years)*100:.1f}%)"
                )

    def test_calculate_depreciation_variance(self):
        """Test depreciation variance calculation with realistic scenarios"""
        # Test with regular depreciation pattern
        result = self.calculator._calculate_depreciation_variance(
            self.mock_depreciation_data,
            10000.0,  # acquisition cost
            5  # life years
        )
        self.assertIsInstance(result, float)
        
        # Expected annual = 10000/5 = 2000
        # Actual annual = 150 * 12 = 1800
        # Variance = ((1800 - 2000) / 2000) * 100 = -10%
        self.assertAlmostEqual(result, -10.0, places=2)
        
        # Test with empty depreciation data
        empty_result = self.calculator._calculate_depreciation_variance(
            pd.DataFrame(),
            10000.0,
            5
        )
        self.assertEqual(empty_result, 0.0)

    def test_determine_utilization_status(self):
        """Test utilization status determination"""
        test_cases = [
            ('IN_SERVICE', 'ACTIVE'),
            ('RETIRED', 'INACTIVE'),
            ('DISPOSED', 'TERMINATED'),
            ('UNKNOWN', 'UNKNOWN')
        ]

        for status, expected in test_cases:
            with self.subTest(status=status):
                result = self.calculator._determine_utilization_status(status)
                self.assertEqual(result, expected)

    def test_invalid_asset_id(self):
        """Test handling of invalid asset ID"""
        self.calculator.db.execute_query.return_value = pd.DataFrame()

        with self.assertRaises(ValueError):
            self.calculator.calculate_asset_depreciation(999999)

        with self.assertRaises(ValueError):
            self.calculator.get_asset_lifecycle_metrics(999999)

    def test_invalid_depreciation_method(self):
        """Test handling of invalid depreciation method"""
        with self.assertRaises(ValueError):
            self.calculator._calculate_annual_depreciation(
                10000.0, 1000.0, 5, 'INVALID_METHOD'
            )

if __name__ == '__main__':
    unittest.main()