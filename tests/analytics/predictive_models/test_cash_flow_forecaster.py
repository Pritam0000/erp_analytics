# tests/analytics/predictive_models/test_cash_flow_forecaster.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.predictive_models.cash_flow_forecaster import CashFlowForecaster

class TestCashFlowForecaster(unittest.TestCase):
    """Test cases for CashFlowForecaster class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.data_loader = Mock()
        self.forecaster = CashFlowForecaster(self.data_loader)
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.lookback_days = 365
        self.forecast_horizon = 30
        
        # Set up mock data
        self.mock_gl_data = self._create_mock_gl_data()
        self.mock_ap_data = self._create_mock_ap_data()
        self.mock_cash_flows = self._create_mock_cash_flows()
        
        # Configure mock data loader
        self.data_loader.data = {
            'gl_journals': self.mock_gl_data['journals'],
            'gl_journal_lines': self.mock_gl_data['lines'],
            'gl_coa': self.mock_gl_data['coa'],
            'ap_invoices': self.mock_ap_data['invoices'],
            'ap_payments': self.mock_ap_data['payments']
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
                lines_data.append({
                    'journal_id': journal.journal_id,
                    'line_id': len(lines_data) + 1,
                    'account_id': np.random.randint(1000, 2000),
                    'debit_amount': np.random.uniform(1000, 10000) if np.random.random() > 0.5 else 0,
                    'credit_amount': np.random.uniform(1000, 10000) if np.random.random() <= 0.5 else 0
                })
        
        lines = pd.DataFrame(lines_data)
        
        # Create chart of accounts
        coa_data = []
        for account_id in range(1000, 2000):
            coa_data.append({
                'account_id': account_id,
                'account_type': np.random.choice(['ASSET', 'LIABILITY', 'EQUITY', 'REVENUE', 'EXPENSE']),
                'account_category': np.random.choice(['OPERATING', 'INVESTING', 'FINANCING']),
                'account_name': f'Account {account_id}'
            })
        
        coa = pd.DataFrame(coa_data)
        
        return {
            'journals': journals,
            'lines': lines,
            'coa': coa
        }

    def _create_mock_ap_data(self):
        """Create mock AP transaction data"""
        # Create invoices
        invoices_data = []
        for i in range(100):
            invoice_date = self.test_date - timedelta(days=np.random.randint(0, self.lookback_days))
            invoices_data.append({
                'invoice_id': i + 1,
                'supplier_id': np.random.randint(1, 20),
                'invoice_date': invoice_date,
                'due_date': invoice_date + timedelta(days=30),
                'amount': np.random.uniform(1000, 5000),
                'status': 'POSTED'
            })
        
        invoices = pd.DataFrame(invoices_data)
        
        # Create payments
        payments_data = []
        for invoice in invoices.itertuples():
            if np.random.random() > 0.2:  # 80% of invoices have payments
                payments_data.append({
                    'payment_id': invoice.invoice_id,
                    'invoice_id': invoice.invoice_id,
                    'payment_date': invoice.due_date + timedelta(days=np.random.randint(-5, 10)),
                    'amount': invoice.amount * np.random.uniform(0.9, 1.0),
                    'status': 'POSTED'
                })
        
        payments = pd.DataFrame(payments_data)
        
        return {
            'invoices': invoices,
            'payments': payments
        }

    def _create_mock_cash_flows(self):
        """Create mock historical cash flow data"""
        dates = pd.date_range(start=self.test_date - timedelta(days=self.lookback_days),
                            end=self.test_date,
                            freq='D')
        
        base_flow = 10000
        trend = np.linspace(0, 5000, len(dates))
        seasonality = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 2000
        noise = np.random.normal(0, 1000, len(dates))
        
        cash_flows = pd.DataFrame({
            'date': dates,
            'operating_cash_flow': base_flow + trend + seasonality + noise,
            'investing_cash_flow': -np.random.uniform(1000, 3000, len(dates)),
            'financing_cash_flow': np.random.uniform(-2000, 2000, len(dates))
        })
        
        cash_flows['net_cash_flow'] = (cash_flows['operating_cash_flow'] + 
                                     cash_flows['investing_cash_flow'] + 
                                     cash_flows['financing_cash_flow'])
        
        return cash_flows

    def test_prepare_training_data(self):
        """Test training data preparation"""
        # Prepare training data
        training_data = self.forecaster.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Verify structure
        self.assertIsInstance(training_data, pd.DataFrame)
        self.assertGreater(len(training_data), 0)
        self.assertGreater(len(training_data.columns), 0)
        
        # Verify required feature categories are present
        feature_categories = ['gl_features', 'ap_features', 'cash_flow_features', 'time_features']
        for category in feature_categories:
            category_columns = [col for col in training_data.columns if category in col]
            self.assertGreater(len(category_columns), 0)
        
        # Verify target variable
        self.assertIn('target', training_data.columns)
        
        # Verify no missing values
        self.assertFalse(training_data.isnull().any().any())

    def test_train_model(self):
        """Test model training"""
        # Prepare training data
        training_data = self.forecaster.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Train model
        results = self.forecaster.train_model(training_data)
        
        # Verify training results
        self.assertIn('training_metrics', results)
        self.assertIn('testing_metrics', results)
        self.assertIn('feature_importance', results)
        
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

    def test_predict_cash_flow(self):
        """Test cash flow prediction"""
        # Train model first
        training_data = self.forecaster.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(training_data)
        
        # Make predictions
        forecast_results = self.forecaster.predict_cash_flow(
            prediction_date=self.test_date,
            forecast_horizon=self.forecast_horizon
        )
        
        # Verify forecast structure
        self.assertIn('dates', forecast_results)
        self.assertIn('predictions', forecast_results)
        self.assertIn('confidence_intervals', forecast_results)
        
        # Verify forecast length
        self.assertEqual(len(forecast_results['predictions']), self.forecast_horizon)
        self.assertEqual(len(forecast_results['confidence_intervals']), self.forecast_horizon)
        
        # Verify confidence intervals
        for interval in forecast_results['confidence_intervals']:
            self.assertLess(interval[0], interval[1])

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Train model first
        training_data = self.forecaster.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(training_data)
        
        # Create evaluation data
        eval_start = self.test_date
        eval_end = eval_start + timedelta(days=30)
        
        evaluation_data = self.forecaster.prepare_training_data(
            start_date=eval_start,
            end_date=eval_end
        )
        
        # Evaluate model
        eval_metrics = self.forecaster.evaluate_model(evaluation_data)
        
        # Verify metrics
        self.assertIn('mean_error', eval_metrics)
        self.assertIn('std_error', eval_metrics)
        self.assertIn('max_error', eval_metrics)
        self.assertIn('evaluation_size', eval_metrics)

    def test_analyze_prediction_errors(self):
        """Test prediction error analysis"""
        # Create test data
        actual_values = pd.Series(np.random.normal(10000, 1000, 30))
        predictions = actual_values + np.random.normal(0, 500, 30)
        
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
        training_data = self.forecaster.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(training_data)
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            self.forecaster.save_model(tmp_dir)
            
            # Create new forecaster instance
            new_forecaster = CashFlowForecaster(self.data_loader)
            
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

    def test_get_forecast_summary(self):
        """Test forecast summary generation"""
        # Train model and make predictions first
        training_data = self.forecaster.prepare_training_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.forecaster.train_model(training_data)
        
        forecast_results = self.forecaster.predict_cash_flow(
            prediction_date=self.test_date,
            forecast_horizon=self.forecast_horizon
        )
        
        # Get summary
        summary = self.forecaster.get_forecast_summary(forecast_results)
        
        # Verify summary structure
        self.assertIn('forecast_period', summary)
        self.assertIn('forecast_statistics', summary)
        self.assertIn('confidence_intervals', summary)
        self.assertIn('model_performance', summary)
        
        # Verify forecast statistics
        stats = summary['forecast_statistics']
        self.assertIn('mean_forecast', stats)
        self.assertIn('min_forecast', stats)
        self.assertIn('max_forecast', stats)
        self.assertIn('std_forecast', stats)

if __name__ == '__main__':
    unittest.main()