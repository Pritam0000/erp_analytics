# tests/analytics/anomaly_detection/test_payment_pattern_analyzer.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analytics.anomaly_detection.payment_pattern_analyzer import PaymentPatternAnalyzer

class TestPaymentPatternAnalyzer(unittest.TestCase):
    """Test cases for PaymentPatternAnalyzer class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.analyzer = PaymentPatternAnalyzer(Mock())
        # Mock database connection and logger
        self.analyzer.logger = Mock()
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.start_date = self.test_date - timedelta(days=365)
        self.end_date = self.test_date
        
        # Generate mock data
        self.mock_ap_data = self.generate_mock_ap_data()
        self.mock_supplier_data = self.generate_mock_supplier_data()
        self.mock_payment_history = self.generate_mock_payment_history()

    def generate_mock_ap_data(self) -> pd.DataFrame:
        """Generate mock AP transaction data with realistic patterns"""
        n_transactions = 1000
        
        # Create base data
        data = pd.DataFrame({
            'invoice_id': range(1, n_transactions + 1),
            'supplier_id': np.random.choice(range(101, 111), size=n_transactions),
            'invoice_date': pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')[:n_transactions],
            'due_date': None,
            'amount_x': np.random.uniform(1000, 10000, size=n_transactions),
            'payment_date': None,
            'amount_y': None,
            'payment_method': np.random.choice(['CHECK', 'ACH', 'WIRE'], size=n_transactions)
        })
        
        # Set due dates (30-60 days from invoice date)
        data['due_date'] = data['invoice_date'] + pd.Timedelta(days=np.random.randint(30, 61, size=n_transactions))
        
        # Add payment information for some invoices
        paid_mask = np.random.choice([True, False], size=n_transactions, p=[0.8, 0.2])
        data.loc[paid_mask, 'payment_date'] = data.loc[paid_mask, 'due_date'] + \
            pd.Timedelta(days=np.random.normal(0, 5, size=paid_mask.sum()))
        data.loc[paid_mask, 'amount_y'] = data.loc[paid_mask, 'amount_x'] * \
            np.random.normal(1, 0.01, size=paid_mask.sum())  # Slight variations in payment amounts
            
        # Add some anomalous patterns
        anomaly_indices = np.random.choice(n_transactions, size=50, replace=False)
        
        # Late payments
        data.loc[anomaly_indices[:20], 'payment_date'] = data.loc[anomaly_indices[:20], 'due_date'] + \
            pd.Timedelta(days=30)
            
        # Early payments
        data.loc[anomaly_indices[20:30], 'payment_date'] = data.loc[anomaly_indices[20:30], 'invoice_date'] + \
            pd.Timedelta(days=1)
            
        # Overpayments
        data.loc[anomaly_indices[30:40], 'amount_y'] = data.loc[anomaly_indices[30:40], 'amount_x'] * 1.5
        
        # Underpayments
        data.loc[anomaly_indices[40:], 'amount_y'] = data.loc[anomaly_indices[40:], 'amount_x'] * 0.5
        
        return data

    def generate_mock_supplier_data(self) -> pd.DataFrame:
        """Generate mock supplier master data"""
        n_suppliers = 10
        
        data = pd.DataFrame({
            'supplier_id': range(101, 101 + n_suppliers),
            'supplier_number': [f'S{i:04d}' for i in range(1, n_suppliers + 1)],
            'supplier_name': [f'Supplier {i}' for i in range(1, n_suppliers + 1)],
            'payment_terms': np.random.choice(['NET30', 'NET45', 'NET60'], size=n_suppliers),
            'status': 'ACTIVE',
            'created_date': pd.date_range(start='2022-01-01', periods=n_suppliers, freq='M')
        })
        
        return data

    def generate_mock_payment_history(self) -> pd.DataFrame:
        """Generate mock payment history data"""
        return self.mock_ap_data[
            ~self.mock_ap_data['payment_date'].isna()
        ].copy()

    def test_prepare_pattern_data(self):
        """Test preparation of payment pattern data"""
        # Mock data loading functions
        self.analyzer._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.analyzer._get_supplier_data = Mock(return_value=self.mock_supplier_data)
        self.analyzer._get_payment_history = Mock(return_value=self.mock_payment_history)
        
        # Test data preparation
        pattern_data = self.analyzer.prepare_pattern_data(
            self.start_date,
            self.end_date
        )
        
        # Verify structure
        self.assertIsInstance(pattern_data, pd.DataFrame)
        self.assertGreater(len(pattern_data.columns), 0)
        
        # Verify feature categories
        required_prefixes = ['timing_', 'amount_', 'pattern_']
        for prefix in required_prefixes:
            self.assertTrue(
                any(col.startswith(prefix) for col in pattern_data.columns),
                f"Missing features with prefix: {prefix}"
            )
            
        # Verify specific features
        key_features = [
            'timing_regularity',
            'amount_variability',
            'payment_irregularity',
            'pattern_complexity'
        ]
        for feature in key_features:
            self.assertIn(feature, pattern_data.columns)

    def test_analyze_payment_patterns(self):
        """Test payment pattern analysis"""
        # Mock data loading functions
        self.analyzer._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.analyzer._get_supplier_data = Mock(return_value=self.mock_supplier_data)
        self.analyzer._get_payment_history = Mock(return_value=self.mock_payment_history)
        
        # Prepare pattern data
        pattern_data = self.analyzer.prepare_pattern_data(
            self.start_date,
            self.end_date
        )
        
        # Test pattern analysis
        results = self.analyzer.analyze_payment_patterns(pattern_data)
        
        # Verify structure
        self.assertIn('cluster_metrics', results)
        self.assertIn('pattern_profiles', results)
        self.assertIn('analysis_params', results)
        
        # Verify cluster metrics
        clusters = results['cluster_metrics']
        self.assertIn('n_clusters', clusters)
        self.assertIn('cluster_sizes', clusters)
        self.assertIn('cluster_stats', clusters)
        
        # Verify pattern profiles
        profiles = results['pattern_profiles']
        self.assertGreater(len(profiles), 0)
        for profile in profiles.values():
            self.assertIn('characteristics', profile)
            self.assertIn('size', profile)

    def test_analyze_supplier_patterns(self):
        """Test supplier-specific pattern analysis"""
        # Mock data loading functions
        self.analyzer._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.analyzer._get_supplier_data = Mock(return_value=self.mock_supplier_data)
        self.analyzer._get_payment_history = Mock(return_value=self.mock_payment_history)
        
        # Test supplier analysis
        results = self.analyzer.analyze_supplier_patterns(101)  # Test first supplier
        
        # Verify structure
        self.assertIn('supplier_id', results)
        self.assertIn('pattern_metrics', results)
        self.assertIn('pattern_changes', results)
        self.assertIn('risk_metrics', results)
        
        # Verify pattern metrics
        metrics = results['pattern_metrics']
        self.assertIn('timing_metrics', metrics)
        self.assertIn('amount_metrics', metrics)
        self.assertIn('behavior_metrics', metrics)
        
        # Verify risk assessment
        risk = results['risk_metrics']
        self.assertIn('timing_risk', risk)
        self.assertIn('amount_risk', risk)
        self.assertIn('behavior_risk', risk)
        self.assertIn('composite_risk', risk)

    # tests/analytics/anomaly_detection/test_payment_pattern_analyzer.py (continued)

    def test_pattern_clustering(self):
        """Test pattern clustering functionality"""
        # Prepare test data with known patterns
        test_data = pd.DataFrame({
            'timing_regularity': [0.9, 0.8, 0.2, 0.1, 0.85] * 20,
            'amount_regularity': [0.85, 0.9, 0.15, 0.2, 0.9] * 20,
            'payment_delay': [2, 1, 30, 45, 3] * 20,
            'amount_variability': [0.1, 0.15, 0.8, 0.9, 0.12] * 20
        })
        
        # Train DBSCAN
        eps = self.analyzer._estimate_eps(
            self.analyzer.scaler.fit_transform(test_data)
        )
        
        # Initialize and fit DBSCAN
        self.analyzer.dbscan_model.set_params(eps=eps)
        cluster_labels = self.analyzer.dbscan_model.fit_predict(
            self.analyzer.scaler.transform(test_data)
        )
        
        # Get unique clusters (excluding noise points marked as -1)
        unique_clusters = len(set(cluster_labels[cluster_labels != -1]))
        
        # There should be at least 2 clusters (good and bad patterns)
        self.assertGreaterEqual(unique_clusters, 2)
        
        # Calculate cluster statistics
        cluster_stats = self.analyzer._calculate_cluster_statistics(
            test_data, cluster_labels
        )
        
        # Verify cluster statistics
        self.assertEqual(len(cluster_stats), unique_clusters)
        for stats in cluster_stats.values():
            self.assertIn('size', stats)
            self.assertIn('centroid', stats)
            self.assertIn('std', stats)
            self.assertIn('feature_importance', stats)

    def test_feature_contributions(self):
        """Test calculation of feature contributions to patterns"""
        # Create test data with known anomalies
        test_data = pd.DataFrame({
            'timing_regularity': [0.9] * 45 + [0.1] * 5,
            'amount_regularity': [0.85] * 45 + [0.2] * 5,
            'payment_delay': [2] * 45 + [30] * 5,
            'amount_variability': [0.1] * 45 + [0.8] * 5
        })
        
        # Calculate anomaly scores
        anomaly_scores = np.zeros(len(test_data))
        anomaly_scores[-5:] = 1  # Mark last 5 records as anomalies
        
        # Calculate feature contributions
        contributions = self.analyzer._calculate_feature_contributions(
            test_data,
            anomaly_scores
        )
        
        # Verify contributions
        self.assertGreater(len(contributions), 0)
        self.assertTrue(all(0 <= score <= 1 for score in contributions.values()))
        
        # Features with high variability should have higher contributions
        timing_contrib = contributions.get('timing_regularity', 0)
        amount_contrib = contributions.get('amount_regularity', 0)
        self.assertGreater(timing_contrib + amount_contrib, 0)

    def test_detect_pattern_changes(self):
        """Test detection of pattern changes"""
        # Create historical and recent data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        historical_data = pd.DataFrame({
            'payment_delay': [5] * 180,
            'amount': [1000] * 180,
            'payment_date': dates[:180],
            'days_to_pay': [30] * 180,
            'payment_ratio': [1.0] * 180
        })
        
        recent_data = pd.DataFrame({
            'payment_delay': [15] * 20,  # Increased delays
            'amount': [2000] * 20,      # Higher amounts
            'payment_date': dates[180:200],
            'days_to_pay': [45] * 20,   # Longer payment cycle
            'payment_ratio': [0.8] * 20  # Underpayments
        })
        
        # Detect changes
        changes = self.analyzer._detect_pattern_changes(
            pd.concat([historical_data, recent_data])
        )
        
        # Verify change detection
        self.assertIn('timing_changes', changes)
        self.assertIn('amount_changes', changes)
        self.assertIn('pattern_changes', changes)
        
        # Verify specific changes
        timing_changes = changes['timing_changes']
        self.assertGreater(timing_changes['payment_delay_change']['mean_change'], 0)
        self.assertGreater(timing_changes['days_to_pay_change']['mean_change'], 0)
        
        amount_changes = changes['amount_changes']
        self.assertGreater(amount_changes['amount_change']['mean_change'], 0)
        self.assertLess(amount_changes['payment_ratio_change']['mean_change'], 0)

    def test_calculate_pattern_complexity(self):
        """Test calculation of pattern complexity"""
        # Create test data with varying complexity
        simple_pattern = pd.DataFrame({
            'payment_delay': [30] * 10,
            'amount': [1000] * 10,
            'payment_ratio': [1.0] * 10,
            'payment_method': ['ACH'] * 10
        })
        
        complex_pattern = pd.DataFrame({
            'payment_delay': list(range(0, 50, 5)),
            'amount': np.random.uniform(500, 1500, 10),
            'payment_ratio': np.random.normal(1.0, 0.1, 10),
            'payment_method': np.random.choice(['ACH', 'CHECK', 'WIRE'], 10)
        })
        
        # Calculate complexity scores
        simple_score = self.analyzer._calculate_pattern_complexity(simple_pattern)
        complex_score = self.analyzer._calculate_pattern_complexity(complex_pattern)
        
        # Verify scores
        self.assertGreater(complex_score, simple_score)
        self.assertLess(simple_score, 0.3)  # Simple patterns should have low complexity
        self.assertGreater(complex_score, 0.5)  # Complex patterns should have high complexity

    def test_calculate_supplier_risk(self):
        """Test calculation of supplier risk metrics"""
        # Create test pattern metrics
        pattern_metrics = {
            'timing_metrics': {
                'avg_payment_delay': 30,
                'payment_delay_variability': 0.5,
                'late_payment_tendency': 60
            },
            'amount_metrics': {
                'amount_variability': 0.4,
                'payment_ratio_consistency': 0.7,
                'overpayment_frequency': 10
            },
            'behavior_metrics': {
                'consistency': {
                    'timing_consistency': 0.6,
                    'amount_consistency': 0.7
                },
                'complexity': {
                    'pattern_complexity': 0.4
                }
            }
        }
        
        # Create test pattern changes
        pattern_changes = {
            'timing_changes': {
                'payment_delay_change': {'mean_change': 20},
                'days_to_pay_change': {'mean_change': 15}
            },
            'amount_changes': {
                'amount_change': {'mean_change': 30},
                'payment_ratio_change': {'mean_change': -0.1}
            }
        }
        
        # Calculate risk
        risk_metrics = self.analyzer._calculate_supplier_risk(
            pattern_metrics,
            pattern_changes
        )
        
        # Verify risk metrics
        self.assertIn('timing_risk', risk_metrics)
        self.assertIn('amount_risk', risk_metrics)
        self.assertIn('behavior_risk', risk_metrics)
        self.assertIn('composite_risk', risk_metrics)
        
        # Verify risk levels
        composite_risk = risk_metrics['composite_risk']
        self.assertIn('risk_score', composite_risk)
        self.assertIn('risk_level', composite_risk)
        
        # High delays and changes should result in higher risk
        self.assertGreater(composite_risk['risk_score'], 0.5)
        self.assertEqual(composite_risk['risk_level'], 'HIGH')

    def test_save_and_load_analyzer(self):
        """Test saving and loading analyzer state"""
        # Mock data loading functions
        self.analyzer._get_ap_data = Mock(return_value=self.mock_ap_data)
        self.analyzer._get_supplier_data = Mock(return_value=self.mock_supplier_data)
        self.analyzer._get_payment_history = Mock(return_value=self.mock_payment_history)
        
        # Prepare and analyze data
        pattern_data = self.analyzer.prepare_pattern_data(
            self.start_date,
            self.end_date
        )
        
        self.analyzer.analyze_payment_patterns(pattern_data)
        
        # Save analyzer state
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'analyzer_state.joblib')
            self.analyzer.save_analyzer(save_path)
            
            # Create new analyzer and load state
            new_analyzer = PaymentPatternAnalyzer(Mock())
            new_analyzer.load_analyzer(save_path)
            
            # Verify loaded state
            self.assertEqual(
                self.analyzer.feature_columns,
                new_analyzer.feature_columns
            )
            self.assertEqual(
                self.analyzer.model_metrics,
                new_analyzer.model_metrics
            )
            
            # Verify model parameters
            self.assertEqual(
                self.analyzer.dbscan_model.get_params(),
                new_analyzer.dbscan_model.get_params()
            )

if __name__ == '__main__':
    unittest.main()