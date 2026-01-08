# tests/analytics/predictive_models/test_depreciation_forecaster.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.predictive_models.depreciation_forecaster import DepreciationForecaster

class TestDepreciationForecaster(unittest.TestCase):
    """Test cases for DepreciationForecaster class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.data_loader = Mock()
        self.forecaster = DepreciationForecaster(self.data_loader)
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.lookback_days = 365
        self.forecast_horizon = 12
        
        # Set up mock data
        self.mock_fa_data = self._create_mock_fa_data()
        self.mock_gl_data = self._create_mock_gl_data()
        self.mock_depreciation_history = self._create_mock_depreciation_history()
        
        # Configure mock data loader
        self.data_loader.data = {
            'fa_assets': self.mock_fa_data['assets'],
            'fa_categories': self.mock_fa_data['categories'],
            'fa_depreciation': self.mock_fa_data['depreciation'],
            'gl_journals': self.mock_gl_data['journals'],
            'gl_journal_lines': self.mock_gl_data['lines'],
            'gl_coa': self.mock_gl_data['coa']
        }

    def _create_mock_fa_data(self):
        """Create mock fixed asset data"""
        # Create assets
        assets_data = []
        for i in range(50):
            acquisition_date = self.test_date - timedelta(days=np.random.randint(0, self.lookback_days*2))
            acquisition_cost = np.random.uniform(10000, 100000)
            life_years = np.random.randint(3, 10)
            
            assets_data.append({
                'asset_id': i + 1,
                'asset_number': f'FA{i+1:04}',
                'asset_name': f'Test Asset {i+1}',
                'category_id': np.random.randint(1, 6),
                'acquisition_date': acquisition_date,
                'acquisition_cost': acquisition_cost,
                'salvage_value': acquisition_cost * 0.1,
                'life_years': life_years,
                'status': 'IN_SERVICE'
            })
        
        assets = pd.DataFrame(assets_data)
        
        # Create categories
        categories_data = [
            {
                'category_id': i + 1,
                'category_code': f'CAT{i+1}',
                'category_name': f'Category {i+1}',
                'depreciation_method': method,
                'life_years': np.random.randint(3, 10)
            }
            for i, method in enumerate(['STRAIGHT_LINE', 'DECLINING_BALANCE', 
                                     'DOUBLE_DECLINING', 'SUM_OF_YEARS_DIGITS', 
                                     'UNITS_OF_PRODUCTION'])
        ]
        
        categories = pd.DataFrame(categories_data)
        
        # Create depreciation history
        depreciation_data = []
        for asset in assets_data:
            start_date = asset['acquisition_date']
            end_date = min(self.test_date, 
                         start_date + timedelta(days=365*asset['life_years']))
            
            # Generate monthly depreciation entries
            current_date = start_date
            accumulated_depreciation = 0
            while current_date <= end_date:
                monthly_depreciation = (
                    (asset['acquisition_cost'] - asset['salvage_value']) / 
                    (asset['life_years'] * 12)
                )
                
                # Add some random variation
                monthly_depreciation *= np.random.uniform(0.95, 1.05)
                accumulated_depreciation += monthly_depreciation
                
                depreciation_data.append({
                    'depreciation_id': len(depreciation_data) + 1,
                    'asset_id': asset['asset_id'],
                    'period_date': current_date,
                    'depreciation_amount': monthly_depreciation,
                    'accumulated_depreciation': accumulated_depreciation,
                    'book_value': asset['acquisition_cost'] - accumulated_depreciation
                })
                
                current_date += timedelta(days=30)
        
        depreciation = pd.DataFrame(depreciation_data)
        
        return {
            'assets': assets,
            'categories': categories,
            'depreciation': depreciation
        }

    def _create_mock_gl_data(self):
        """Create mock GL transaction data"""
        # Create journals
        journals_data = []
        dates = pd.date_range(start=self.test_date - timedelta(days=self.lookback_days),
                            end=self.test_date, freq='D')
        
        for i, date in enumerate(dates):
            journals_data.append({
                'journal_id': i + 1,
                'journal_date': date,
                'journal_number': f'JE{i+1:04}',
                'status': 'POSTED'
            })
        
        journals = pd.DataFrame(journals_data)
        
        # Create journal lines
        lines_data = []
        for journal in journals_data:
            # Add depreciation entries
            lines_data.append({
                'journal_id': journal['journal_id'],
                'line_id': len(lines_data) + 1,
                'account_id': 1001,  # Depreciation expense
                'debit_amount': np.random.uniform(1000, 5000),
                'credit_amount': 0
            })
            lines_data.append({
                'journal_id': journal['journal_id'],
                'line_id': len(lines_data) + 1,
                'account_id': 1002,  # Accumulated depreciation
                'debit_amount': 0,
                'credit_amount': lines_data[-1]['debit_amount']
            })
        
        lines = pd.DataFrame(lines_data)
        
        # Create chart of accounts
        coa_data = [
            {
                'account_id': 1001,
                'account_type': 'EXPENSE',
                'account_category': 'DEPRECIATION',
                'account_name': 'Depreciation Expense'
            },
            {
                'account_id': 1002,
                'account_type': 'CONTRA_ASSET',
                'account_category': 'ACCUMULATED_DEPRECIATION',
                'account_name': 'Accumulated Depreciation'
            }
        ]
        
        coa = pd.DataFrame(coa_data)
        
        return {
            'journals': journals,
            'lines': lines,
            'coa': coa
        }

    def _create_mock_depreciation_history(self):
        """Create mock depreciation history data"""
        return self.mock_fa_data['depreciation']

    def test_prepare_pattern_data(self):
        """Test preparation of pattern data"""
        pattern_data = self.forecaster.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Verify structure
        self.assertIsInstance(pattern_data, pd.DataFrame)
        self.assertGreater(len(pattern_data), 0)
        self.assertGreater(len(pattern_data.columns), 0)
        
        # Verify feature categories
        required_features = ['asset_features', 'gl_features', 
                           'depreciation_features', 'time_features']
        
        for feature_type in required_features:
            matching_cols = [col for col in pattern_data.columns 
                           if feature_type in col]
            self.assertGreater(len(matching_cols), 0)
        
        # Verify no missing values
        self.assertFalse(pattern_data.isnull().any().any())

    def test_train_model(self):
        """Test model training"""
        # Prepare training data
        pattern_data = self.forecaster.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Train model
        results = self.forecaster.train_model(pattern_data)
        
        # Verify results structure
        self.assertIn('training_metrics', results)
        self.assertIn('testing_metrics', results)
        self.assertIn('feature_importance', results)
        self.assertIn('prediction_intervals', results)
        
        # Verify metrics
        metrics = results['training_metrics']
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        
        # Verify feature importance
        self.assertEqual(
            len(results['feature_importance']),
            len(self.forecaster.feature_columns)
        )

    def test_predict_depreciation(self):
        """Test depreciation prediction"""
        # Train model first
        pattern_data = self.forecaster.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(pattern_data)
        
        # Make predictions
        asset_data = self.mock_fa_data['assets'].head()
        forecast_results = self.forecaster.predict_depreciation(
            asset_data,
            forecast_periods=self.forecast_horizon
        )
        
        # Verify forecast structure
        self.assertIn('forecasts', forecast_results)
        self.assertIn('model_metrics', forecast_results)
        self.assertIn('feature_importance', forecast_results)
        
        # Verify individual forecasts
        forecasts = forecast_results['forecasts']
        self.assertEqual(len(forecasts), len(asset_data))
        
        for forecast in forecasts:
            self.assertIn('asset_id', forecast)
            self.assertIn('asset_name', forecast)
            self.assertIn('forecast', forecast)
            self.assertEqual(
                len(forecast['forecast']['predictions']),
                self.forecast_horizon
            )

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Train model first
        pattern_data = self.forecaster.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(pattern_data)
        
        # Create evaluation data
        eval_start = self.test_date
        eval_end = eval_start + timedelta(days=30)
        
        evaluation_data = self.forecaster.prepare_pattern_data(
            start_date=eval_start,
            end_date=eval_end
        )
        
        # Evaluate model
        eval_metrics = self.forecaster.evaluate_model(evaluation_data)
        
        # Verify metrics structure
        self.assertIn('mean_error', eval_metrics)
        self.assertIn('std_error', eval_metrics)
        self.assertIn('max_error', eval_metrics)
        self.assertIn('evaluation_size', eval_metrics)

    def test_analyze_prediction_errors(self):
        """Test prediction error analysis"""
        # Create test data
        actual_values = pd.Series(np.random.normal(1000, 100, 30))
        predictions = actual_values + np.random.normal(0, 50, 30)
        
        # Analyze errors
        error_analysis = self.forecaster.analyze_prediction_errors(
            actual_values,
            predictions
        )
        
        # Verify analysis structure
        self.assertIn('error_statistics', error_analysis)
        self.assertIn('error_distribution', error_analysis)
        self.assertIn('error_classification', error_analysis)
        
        # Verify statistics
        stats = error_analysis['error_statistics']
        self.assertIn('mean_error', stats)
        self.assertIn('median_error', stats)
        self.assertIn('std_error', stats)
        self.assertIn('mean_absolute_error', stats)
        self.assertIn('mean_percentage_error', stats)

    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Train model first
        pattern_data = self.forecaster.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(pattern_data)
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            self.forecaster.save_model(tmp_dir)
            
            # Create new forecaster instance
            new_forecaster = DepreciationForecaster(self.data_loader)
            
            # Load model
            new_forecaster.load_model(tmp_dir)
            
            # Verify loaded model has same attributes
            self.assertEqual(
                self.forecaster.feature_columns,
                new_forecaster.feature_columns
            )
            self.assertEqual(
                self.forecaster.training_period,
                new_forecaster.training_period
            )
            self.assertEqual(
                self.forecaster.model_metrics,
                new_forecaster.model_metrics
            )

    def test_get_model_summary(self):
        """Test model summary generation"""
        # Train model first
        pattern_data = self.forecaster.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(pattern_data)
        
        # Get summary
        summary = self.forecaster.get_model_summary()
        
        # Verify summary structure
        self.assertIn('model_info', summary)
        self.assertIn('performance_metrics', summary)
        self.assertIn('top_features', summary)
        self.assertIn('training_data_info', summary)

if __name__ == '__main__':
    unittest.main()