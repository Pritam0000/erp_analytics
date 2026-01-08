# tests/analytics/predictive_models/test_balance_predictor.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.predictive_models.balance_predictor import BalancePredictor

class TestBalancePredictor(unittest.TestCase):
    """Test cases for BalancePredictor class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.data_loader = Mock()
        self.predictor = BalancePredictor(self.data_loader)
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.lookback_days = 365
        self.forecast_horizon = 30
        
        # Set up mock data
        self.mock_gl_data = self._create_mock_gl_data()
        self.mock_balance_history = self._create_mock_balance_history()
        
        # Configure mock data loader
        self.data_loader.data = {
            'gl_journals': self.mock_gl_data['journals'],
            'gl_journal_lines': self.mock_gl_data['lines'],
            'gl_coa': self.mock_gl_data['coa']
        }

    def _create_mock_gl_data(self):
        """Create mock GL transaction data"""
        # Create date range
        dates = pd.date_range(start=self.test_date - timedelta(days=self.lookback_days),
                            end=self.test_date,
                            freq='D')
        
        # Create journals
        journals = pd.DataFrame({
            'journal_id': range(1, len(dates) + 1),
            'journal_date': dates,
            'journal_number': [f'JE{i:04}' for i in range(1, len(dates) + 1)],
            'status': 'POSTED'
        })
        
        # Create journal lines
        lines_data = []
        for journal in journals.itertuples():
            # Add multiple lines per journal
            for _ in range(np.random.randint(2, 5)):
                account_id = np.random.randint(1000, 2000)
                amount = np.random.uniform(1000, 10000)
                is_debit = np.random.random() > 0.5
                
                lines_data.append({
                    'journal_id': journal.journal_id,
                    'line_id': len(lines_data) + 1,
                    'account_id': account_id,
                    'debit_amount': amount if is_debit else 0,
                    'credit_amount': amount if not is_debit else 0,
                    'description': f'Test transaction {len(lines_data) + 1}'
                })
        
        lines = pd.DataFrame(lines_data)
        
        # Create chart of accounts
        coa_data = []
        for account_id in range(1000, 2000):
            account_type = np.random.choice([
                'ASSET', 'LIABILITY', 'EQUITY', 'REVENUE', 'EXPENSE'
            ])
            account_category = np.random.choice([
                'CURRENT', 'NON_CURRENT', 'OPERATING', 'FINANCING'
            ])
            
            coa_data.append({
                'account_id': account_id,
                'account_number': f'A{account_id}',
                'account_name': f'Account {account_id}',
                'account_type': account_type,
                'account_category': account_category,
                'parent_account_id': np.random.randint(1000, 2000) if np.random.random() > 0.8 else None,
                'is_active': 'Y'
            })
        
        coa = pd.DataFrame(coa_data)
        
        return {
            'journals': journals,
            'lines': lines,
            'coa': coa
        }

    def _create_mock_balance_history(self):
        """Create mock balance history data"""
        dates = pd.date_range(start=self.test_date - timedelta(days=self.lookback_days),
                            end=self.test_date,
                            freq='D')
        
        # Create base balance trend with seasonality and noise
        base_balance = 100000
        trend = np.linspace(0, 50000, len(dates))
        seasonality = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 10000
        noise = np.random.normal(0, 5000, len(dates))
        
        balance_data = pd.DataFrame({
            'date': dates,
            'balance': base_balance + trend + seasonality + noise,
            'debit_volume': np.random.uniform(5000, 15000, len(dates)),
            'credit_volume': np.random.uniform(5000, 15000, len(dates))
        })
        
        # Add transaction counts
        balance_data['transaction_count'] = np.random.randint(5, 20, len(dates))
        
        return balance_data

    def test_prepare_training_data(self):
        """Test training data preparation"""
        # Prepare training data
        training_data = self.predictor.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date,
            account_types=['ASSET', 'LIABILITY']
        )
        
        # Verify structure
        self.assertIsInstance(training_data, pd.DataFrame)
        self.assertGreater(len(training_data), 0)
        self.assertGreater(len(training_data.columns), 0)
        
        # Verify required feature categories
        feature_categories = [
            'transaction_features',
            'balance_features',
            'account_features',
            'time_features'
        ]
        
        for category in feature_categories:
            matching_cols = [col for col in training_data.columns if category in col]
            self.assertGreater(len(matching_cols), 0)
        
        # Verify target variable exists
        self.assertIn('target', training_data.columns)
        
        # Verify no missing values
        self.assertFalse(training_data.isnull().any().any())

    def test_train_model(self):
        """Test model training"""
        # Prepare training data
        training_data = self.predictor.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Train model
        results = self.predictor.train_model(training_data)
        
        # Verify training results
        self.assertIn('training_metrics', results)
        self.assertIn('testing_metrics', results)
        self.assertIn('feature_importance', results)
        self.assertIn('model_metrics', results)
        
        # Verify metrics
        metrics = results['training_metrics']
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        
        # Verify feature importance
        self.assertEqual(
            len(results['feature_importance']),
            len(self.predictor.feature_columns)
        )

    def test_predict_balances(self):
        """Test balance prediction"""
        # Train model first
        training_data = self.predictor.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.predictor.train_model(training_data)
        
        # Make predictions
        account_ids = [1001, 1002, 1003]
        forecast_results = self.predictor.predict_balances(
            account_ids,
            forecast_horizon=self.forecast_horizon
        )
        
        # Verify forecast structure
        self.assertIn('forecasts', forecast_results)
        self.assertIn('model_metrics', forecast_results)
        self.assertIn('feature_importance', forecast_results)
        
        # Verify forecasts
        forecasts = forecast_results['forecasts']
        self.assertEqual(len(forecasts), len(account_ids))
        
        for forecast in forecasts:
            self.assertIn('account_id', forecast)
            self.assertIn('account_name', forecast)
            self.assertIn('forecast', forecast)
            
            self.assertEqual(
                len(forecast['forecast']['predictions']),
                self.forecast_horizon
            )

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Train model first
        training_data = self.predictor.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.predictor.train_model(training_data)
        
        # Create evaluation data
        eval_start = self.test_date
        eval_end = eval_start + timedelta(days=30)
        
        evaluation_data = self.predictor.prepare_training_data(
            start_date=eval_start,
            end_date=eval_end
        )
        
        # Evaluate model
        eval_metrics = self.predictor.evaluate_model(evaluation_data)
        
        # Verify metrics
        self.assertIn('mean_error', eval_metrics)
        self.assertIn('std_error', eval_metrics)
        self.assertIn('max_error', eval_metrics)
        self.assertIn('evaluation_size', eval_metrics)

    def test_analyze_prediction_errors(self):
        """Test prediction error analysis"""
        # Create test data
        actual_values = pd.Series(np.random.normal(100000, 10000, 30))
        predictions = actual_values + np.random.normal(0, 5000, 30)
        
        # Analyze errors
        error_analysis = self.predictor.analyze_prediction_errors(
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
        training_data = self.predictor.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.predictor.train_model(training_data)
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            self.predictor.save_model(tmp_dir)
            
            # Create new predictor instance
            new_predictor = BalancePredictor(self.data_loader)
            
            # Load model
            new_predictor.load_model(tmp_dir)
            
            # Verify loaded model has same attributes
            self.assertEqual(
                self.predictor.feature_columns,
                new_predictor.feature_columns
            )
            self.assertEqual(
                self.predictor.training_period,
                new_predictor.training_period
            )
            self.assertEqual(
                self.predictor.model_metrics,
                new_predictor.model_metrics
            )

    def test_analyze_balance_patterns(self):
        """Test balance pattern analysis"""
        account_id = 1001
        
        pattern_analysis = self.predictor.analyze_balance_patterns(account_id)
        
        # Verify analysis structure
        self.assertIn('pattern_metrics', pattern_analysis)
        self.assertIn('trend_analysis', pattern_analysis)
        self.assertIn('seasonality_analysis', pattern_analysis)
        self.assertIn('volatility_metrics', pattern_analysis)
        
        # Verify pattern metrics
        metrics = pattern_analysis['pattern_metrics']
        self.assertIn('avg_balance', metrics)
        self.assertIn('std_balance', metrics)
        self.assertIn('min_balance', metrics)
        self.assertIn('max_balance', metrics)
        
        # Verify trend analysis
        trend = pattern_analysis['trend_analysis']
        self.assertIn('trend_direction', trend)
        self.assertIn('trend_strength', trend)
        self.assertIn('trend_significance', trend)

    def test_get_model_summary(self):
        """Test model summary generation"""
        # Train model first
        training_data = self.predictor.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.predictor.train_model(training_data)
        
        # Get summary
        summary = self.predictor.get_model_summary()
        
        # Verify summary structure
        self.assertIn('model_info', summary)
        self.assertIn('performance_metrics', summary)
        self.assertIn('top_features', summary)
        self.assertIn('training_data_info', summary)
        
        # Verify model info
        model_info = summary['model_info']
        self.assertIn('model_type', model_info)
        self.assertIn('n_features', model_info)
        self.assertIn('training_period', model_info)

if __name__ == '__main__':
    unittest.main()