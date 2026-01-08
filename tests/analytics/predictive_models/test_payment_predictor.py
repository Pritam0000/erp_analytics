# tests/analytics/predictive_models/test_payment_predictor.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.predictive_models.payment_predictor import PaymentPredictor

class TestPaymentPredictor(unittest.TestCase):
    """Test cases for PaymentPredictor class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.data_loader = Mock()
        self.predictor = PaymentPredictor(self.data_loader)
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.lookback_days = 365
        
        # Set up mock data
        self.mock_ap_data = self._create_mock_ap_data()
        self.mock_supplier_data = self._create_mock_supplier_data()
        self.mock_payment_history = self._create_mock_payment_history()
        
        # Configure mock data loader
        self.data_loader.data = {
            'ap_invoices': self.mock_ap_data['invoices'],
            'ap_payments': self.mock_ap_data['payments'],
            'ap_suppliers': self.mock_supplier_data
        }

    def _create_mock_ap_data(self):
        """Create mock AP transaction data"""
        # Create invoices
        invoices_data = []
        for i in range(200):
            invoice_date = self.test_date - timedelta(days=np.random.randint(0, self.lookback_days))
            invoices_data.append({
                'invoice_id': i + 1,
                'supplier_id': np.random.randint(1, 20),
                'invoice_number': f'INV{i+1:04}',
                'invoice_date': invoice_date,
                'due_date': invoice_date + timedelta(days=30),
                'amount': np.random.uniform(1000, 5000),
                'status': np.random.choice(['PAID', 'PARTIALLY_PAID', 'UNPAID']),
                'created_date': invoice_date
            })
        
        invoices = pd.DataFrame(invoices_data)
        
        # Create payments
        payments_data = []
        for invoice in invoices[invoices['status'].isin(['PAID', 'PARTIALLY_PAID'])].itertuples():
            payment_date = invoice.due_date + timedelta(days=np.random.randint(-5, 15))
            payments_data.append({
                'payment_id': len(payments_data) + 1,
                'invoice_id': invoice.invoice_id,
                'supplier_id': invoice.supplier_id,
                'payment_date': payment_date,
                'payment_method': np.random.choice(['CHECK', 'ACH', 'WIRE']),
                'amount': invoice.amount * np.random.uniform(0.8, 1.0),
                'status': 'POSTED'
            })
        
        payments = pd.DataFrame(payments_data)
        
        return {
            'invoices': invoices,
            'payments': payments
        }

    def _create_mock_supplier_data(self):
        """Create mock supplier data"""
        suppliers_data = []
        for i in range(20):
            suppliers_data.append({
                'supplier_id': i + 1,
                'supplier_number': f'SUP{i+1:04}',
                'supplier_name': f'Supplier {i+1}',
                'payment_terms': np.random.choice(['NET30', 'NET45', 'NET60']),
                'status': 'ACTIVE',
                'created_date': self.test_date - timedelta(days=np.random.randint(100, 1000))
            })
        
        return pd.DataFrame(suppliers_data)

    def _create_mock_payment_history(self):
        """Create mock payment history data"""
        history_data = []
        for supplier_id in range(1, 21):
            for _ in range(np.random.randint(10, 50)):
                payment_date = self.test_date - timedelta(days=np.random.randint(0, self.lookback_days))
                history_data.append({
                    'supplier_id': supplier_id,
                    'payment_date': payment_date,
                    'amount': np.random.uniform(1000, 5000),
                    'payment_method': np.random.choice(['CHECK', 'ACH', 'WIRE']),
                    'days_to_pay': np.random.randint(-5, 45)
                })
        
        return pd.DataFrame(history_data)

    def test_prepare_pattern_data(self):
        """Test pattern data preparation"""
        # Prepare pattern data
        pattern_data = self.predictor.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Verify structure
        self.assertIsInstance(pattern_data, pd.DataFrame)
        self.assertGreater(len(pattern_data), 0)
        self.assertGreater(len(pattern_data.columns), 0)
        
        # Verify required feature categories are present
        feature_categories = [
            'timing_features', 
            'amount_features',
            'supplier_features',
            'pattern_features'
        ]
        for category in feature_categories:
            category_columns = [col for col in pattern_data.columns if category in col]
            self.assertGreater(len(category_columns), 0)
        
        # Verify no missing values
        self.assertFalse(pattern_data.isnull().any().any())

    def test_train_detector(self):
        """Test payment pattern detector training"""
        # Prepare pattern data
        pattern_data = self.predictor.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Train detector
        results = self.predictor.train_detector(pattern_data)
        
        # Verify training results
        self.assertIn('cluster_metrics', results)
        self.assertIn('anomaly_metrics', results)
        self.assertIn('pattern_profiles', results)
        
        # Verify cluster metrics
        cluster_metrics = results['cluster_metrics']
        self.assertIn('n_clusters', cluster_metrics)
        self.assertIn('cluster_sizes', cluster_metrics)
        self.assertIn('cluster_stats', cluster_metrics)
        
        # Verify anomaly metrics
        anomaly_metrics = results['anomaly_metrics']
        self.assertIn('anomaly_count', anomaly_metrics)
        self.assertIn('anomaly_rate', anomaly_metrics)
        self.assertIn('anomaly_scores', anomaly_metrics)

    def test_analyze_payment_patterns(self):
        """Test payment pattern analysis"""
        # Prepare pattern data
        pattern_data = self.predictor.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Train detector first
        self.predictor.train_detector(pattern_data)
        
        # Analyze patterns
        analysis_results = self.predictor.analyze_payment_patterns(pattern_data)
        
        # Verify analysis structure
        self.assertIn('cluster_metrics', analysis_results)
        self.assertIn('anomaly_metrics', analysis_results)
        self.assertIn('pattern_profiles', analysis_results)
        
        # Verify cluster analysis
        clusters = analysis_results['cluster_metrics']
        self.assertGreater(clusters['n_clusters'], 0)
        self.assertEqual(
            sum(clusters['cluster_sizes'].values()),
            len(pattern_data)
        )
        
        # Verify pattern profiles
        profiles = analysis_results['pattern_profiles']
        for cluster_id in range(clusters['n_clusters']):
            cluster_key = f'cluster_{cluster_id}'
            self.assertIn(cluster_key, profiles)
            self.assertIn('size', profiles[cluster_key])
            self.assertIn('characteristics', profiles[cluster_key])

    def test_analyze_supplier_patterns(self):
        """Test supplier pattern analysis"""
        # Prepare pattern data
        pattern_data = self.predictor.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        
        # Train detector first
        self.predictor.train_detector(pattern_data)
        
        # Analyze supplier patterns
        supplier_id = 1
        analysis = self.predictor.analyze_supplier_patterns(supplier_id)
        
        # Verify analysis structure
        self.assertEqual(analysis['supplier_id'], supplier_id)
        self.assertIn('pattern_metrics', analysis)
        self.assertIn('pattern_changes', analysis)
        self.assertIn('risk_metrics', analysis)
        self.assertIn('recommendations', analysis)
        
        # Verify pattern metrics
        metrics = analysis['pattern_metrics']
        self.assertIn('timing_metrics', metrics)
        self.assertIn('amount_metrics', metrics)
        self.assertIn('behavior_metrics', metrics)
        
        # Verify risk metrics
        risk = analysis['risk_metrics']
        self.assertIn('timing_risk', risk)
        self.assertIn('amount_risk', risk)
        self.assertIn('behavior_risk', risk)
        self.assertIn('composite_risk', risk)

    def test_detect_pattern_changes(self):
        """Test pattern change detection"""
        # Create test payment history
        payment_history = pd.DataFrame({
            'payment_date': pd.date_range(
                start=self.test_date - timedelta(days=180),
                end=self.test_date,
                freq='D'
            ),
            'amount': np.random.uniform(1000, 5000, 181),
            'days_to_pay': np.random.randint(-5, 45, 181)
        })
        
        # Detect changes
        changes = self.predictor._detect_pattern_changes(payment_history)
        
        # Verify changes structure
        self.assertIn('timing_changes', changes)
        self.assertIn('amount_changes', changes)
        self.assertIn('pattern_changes', changes)
        
        # Verify change metrics
        timing_changes = changes['timing_changes']
        self.assertIn('payment_delay_change', timing_changes)
        self.assertIn('days_to_pay_change', timing_changes)
        
        amount_changes = changes['amount_changes']
        self.assertIn('amount_change', amount_changes)
        
        pattern_changes = changes['pattern_changes']
        self.assertIn('regularity_change', pattern_changes)
        self.assertIn('complexity_change', pattern_changes)

    def test_calculate_supplier_risk(self):
        """Test supplier risk calculation"""
        # Create test metrics
        pattern_metrics = {
            'timing_metrics': {
                'avg_payment_delay': 10.5,
                'payment_delay_variability': 0.2,
                'late_payment_rate': 0.15
            },
            'amount_metrics': {
                'amount_variability': 0.3,
                'payment_ratio_consistency': 0.8
            },
            'behavior_metrics': {
                'consistency': {
                    'timing_consistency': 0.7,
                    'amount_consistency': 0.8
                }
            }
        }
        
        pattern_changes = {
            'timing_changes': {
                'payment_delay_change': {'mean_change': 5.0}
            },
            'amount_changes': {
                'amount_change': {'mean_change': 10.0}
            },
            'pattern_changes': {
                'regularity_change': -0.1
            }
        }
        
        # Calculate risk
        risk_metrics = self.predictor._calculate_supplier_risk(
            pattern_metrics,
            pattern_changes
        )
        
        # Verify risk metrics
        self.assertIn('timing_risk', risk_metrics)
        self.assertIn('amount_risk', risk_metrics)
        self.assertIn('behavior_risk', risk_metrics)
        self.assertIn('composite_risk', risk_metrics)
        
        # Verify risk scores
        for risk_type in ['timing_risk', 'amount_risk', 'behavior_risk']:
            risk = risk_metrics[risk_type]
            self.assertGreaterEqual(risk['risk_score'], 0)
            self.assertLessEqual(risk['risk_score'], 1)
            self.assertIn(risk['risk_level'], ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Create test risk metrics
        risk_metrics = {
            'composite_risk': {
                'risk_level': 'HIGH',
                'risk_score': 0.8
            },
            'timing_risk': {
                'risk_level': 'HIGH',
                'risk_score': 0.85
            },
            'amount_risk': {
                'risk_level': 'MEDIUM',
                'risk_score': 0.6
            },
            'behavior_risk': {
                'risk_level': 'HIGH',
                'risk_score': 0.75
            }
        }
        
        # Generate recommendations
        recommendations = self.predictor._generate_recommendations(risk_metrics)
        
        # Verify recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Verify high risk recommendations
        high_risk_keywords = ['immediate', 'review', 'monitor']
        self.assertTrue(
            any(keyword in rec.lower() for rec in recommendations 
                for keyword in high_risk_keywords)
        )

    def test_save_and_load_analyzer(self):
        """Test analyzer saving and loading"""
        # Prepare and train detector first
        pattern_data = self.predictor.prepare_pattern_data(
            start_date=self.test_date - timedelta(days=self.lookback_days),
            end_date=self.test_date
        )
        self.predictor.train_detector(pattern_data)
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save analyzer
            self.predictor.save_analyzer(tmp_dir)
            
            # Create new predictor instance
            new_predictor = PaymentPredictor(self.data_loader)
            
            # Load analyzer
            new_predictor.load_analyzer(tmp_dir)
            
            # Verify loaded analyzer has same attributes
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

if __name__ == '__main__':
    unittest.main()