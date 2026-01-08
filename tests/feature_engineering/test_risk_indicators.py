# tests/feature_engineering/test_risk_indicators.py

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.feature_engineering.risk_indicators.risk_indicators import RiskIndicatorCalculator

class TestRiskIndicatorCalculator(unittest.TestCase):
    """Test cases for RiskIndicatorCalculator class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.calculator = RiskIndicatorCalculator()
        # Mock database connection and logger
        self.calculator.db = Mock()
        self.calculator.logger = Mock()
        
        # Set up test dates
        self.test_date = datetime(2024, 1, 1)
        self.lookback_days = 365
        self.start_date = self.test_date - timedelta(days=self.lookback_days)
        
        # Set up mock payment data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        self.mock_payment_data = pd.DataFrame({
            'supplier_id': [1] * len(dates),
            'total_invoices': [100] * len(dates),
            'defaulted_invoices': [5] * len(dates),
            'avg_days_outstanding': [15.5] * len(dates),
            'max_days_outstanding': [45] * len(dates),
            'total_exposure': [50000.0] * len(dates),
            'total_paid': [48000.0] * len(dates),
            'overdue_30_plus': [10] * len(dates),
            'overdue_90_plus': [2] * len(dates)
        })
        
        # Set up mock account data
        self.mock_account_data = pd.DataFrame({
            'transaction_date': dates,
            'net_change': np.random.normal(0, 1000, size=len(dates)),
            'running_balance': np.cumsum(np.random.normal(0, 1000, size=len(dates))),
            'daily_change': np.random.normal(0, 500, size=len(dates))
        })
        
        # Set up mock transaction data
        self.mock_transaction_data = pd.DataFrame({
            'journal_id': range(1, len(dates) + 1),
            'journal_date': dates,
            'line_count': np.random.randint(2, 10, size=len(dates)),
            'account_count': np.random.randint(1, 5, size=len(dates)),
            'total_debits': np.random.uniform(1000, 10000, size=len(dates)),
            'total_credits': np.random.uniform(1000, 10000, size=len(dates)),
            'max_amount': np.random.uniform(5000, 15000, size=len(dates))
        })
        
        # Add some anomalies to transaction data
        anomaly_indices = np.random.choice(len(dates), size=5, replace=False)
        self.mock_transaction_data.loc[anomaly_indices, 'max_amount'] *= 5
        self.mock_transaction_data.loc[anomaly_indices, 'line_count'] *= 3
        
        # Calculate rolling statistics for transaction data
        self.mock_transaction_data['avg_line_count'] = (
            self.mock_transaction_data['line_count'].rolling(window=30, min_periods=1).mean()
        )
        self.mock_transaction_data['avg_amount'] = (
            self.mock_transaction_data['max_amount'].rolling(window=30, min_periods=1).mean()
        )
        self.mock_transaction_data['stddev_amount'] = (
            self.mock_transaction_data['max_amount'].rolling(window=30, min_periods=1).std()
        )

    def test_analyze_payment_default_risk(self):
        """Test payment default risk analysis"""
        # Set up mock return value
        self.calculator.db.execute_query.return_value = self.mock_payment_data

        # Test with specific supplier
        result = self.calculator.analyze_payment_default_risk(
            supplier_id=1,
            lookback_days=self.lookback_days
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('risk_metrics', result)
        self.assertIn('period', result)

        # Verify risk metrics
        risk_metrics = result['risk_metrics'][0]  # First supplier
        self.assertIn('risk_score', risk_metrics)
        self.assertIn('risk_level', risk_metrics)
        self.assertIn('metrics', risk_metrics)

        # Verify metric calculations
        metrics = risk_metrics['metrics']
        self.assertGreaterEqual(metrics['default_rate'], 0)
        self.assertLessEqual(metrics['default_rate'], 100)
        self.assertGreater(metrics['total_exposure'], 0)

        # Verify risk level is valid
        self.assertIn(risk_metrics['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])

    def test_analyze_payment_default_risk_empty_data(self):
        """Test payment default risk analysis with empty data"""
        # Set up mock to return empty DataFrame
        self.calculator.db.execute_query.return_value = pd.DataFrame()

        result = self.calculator.analyze_payment_default_risk(
            supplier_id=999,
            lookback_days=self.lookback_days
        )

        # Verify handling of empty data
        self.assertEqual(result['status'], 'NO_DATA')
        self.assertIn('period', result)

    def test_analyze_payment_default_risk_all_suppliers(self):
        """Test payment default risk analysis for all suppliers"""
        # Create mock data with two suppliers
        supplier1_data = self.mock_payment_data.copy()
        supplier2_data = self.mock_payment_data.copy()
        supplier2_data['supplier_id'] = 2
        
        # Group by supplier_id to get one row per supplier
        multi_supplier_data = pd.concat([supplier1_data, supplier2_data]).groupby('supplier_id').agg({
            'total_invoices': 'sum',
            'defaulted_invoices': 'sum',
            'avg_days_outstanding': 'mean',
            'max_days_outstanding': 'max',
            'total_exposure': 'sum',
            'total_paid': 'sum',
            'overdue_30_plus': 'sum',
            'overdue_90_plus': 'sum'
        }).reset_index()
        
        self.calculator.db.execute_query.return_value = multi_supplier_data

        result = self.calculator.analyze_payment_default_risk(
            lookback_days=self.lookback_days
        )

        # Verify structure and calculations for multiple suppliers
        self.assertEqual(result['status'], 'ANALYZED')
        self.assertEqual(len(result['risk_metrics']), 2)  # Two suppliers

        # Verify each supplier has valid metrics
        for supplier_metrics in result['risk_metrics']:
            self.assertIn('supplier_id', supplier_metrics)
            self.assertIn('risk_score', supplier_metrics)
            self.assertIn('risk_level', supplier_metrics)
            self.assertIn('metrics', supplier_metrics)

    def test_analyze_account_volatility(self):
        """Test account volatility analysis"""
        # Set up mock return value
        self.calculator.db.execute_query.return_value = self.mock_account_data

        # Test for specific account
        result = self.calculator.analyze_account_volatility(
            account_id=1001,
            period_months=12
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('account_id', result)
        self.assertIn('risk_level', result)
        self.assertIn('stability_metrics', result)
        self.assertIn('period', result)

        # Verify stability metrics
        stability_metrics = result['stability_metrics']
        self.assertIn('volatility_score', stability_metrics)
        self.assertIn('trend_direction', stability_metrics)
        self.assertIn('extreme_movements', stability_metrics)
        self.assertIn('statistics', stability_metrics)

        # Verify statistical calculations
        stats = stability_metrics['statistics']
        self.assertIn('mean_balance', stats)
        self.assertIn('max_balance', stats)
        self.assertIn('min_balance', stats)
        self.assertIn('balance_volatility', stats)
        self.assertIn('daily_volatility', stats)

        # Verify value ranges
        self.assertGreaterEqual(stability_metrics['volatility_score'], 0)
        self.assertLessEqual(stability_metrics['volatility_score'], 100)
        self.assertIn(result['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])

    def test_analyze_account_volatility_empty_data(self):
        """Test account volatility analysis with empty data"""
        # Set up mock to return empty DataFrame
        self.calculator.db.execute_query.return_value = pd.DataFrame()

        result = self.calculator.analyze_account_volatility(
            account_id=999,
            period_months=12
        )

        # Verify handling of empty data
        self.assertEqual(result['status'], 'NO_DATA')
        self.assertEqual(result['account_id'], 999)
        self.assertIn('period_months', result)

    def test_calculate_volatility_score(self):
        """Test volatility score calculation"""
        test_cases = [
            # (balance_vol, daily_vol, extremes, expected_level)
            (500.0, 50.0, 1, 'LOW'),            # Low volatility
            (5000.0, 500.0, 4, 'MEDIUM'),       # Medium volatility
            (20000.0, 2000.0, 8, 'HIGH')        # High volatility
        ]

        for balance_vol, daily_vol, extremes, expected in test_cases:
            with self.subTest(balance_vol=balance_vol, daily_vol=daily_vol):
                score = self.calculator._calculate_volatility_score(
                    balance_volatility=balance_vol,
                    daily_volatility=daily_vol,
                    extreme_movements=extremes
                )
                
                # Verify score is within expected range
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)
                
                # Calculate risk level
                risk_level = self.calculator._assess_volatility_risk(
                    volatility_score=score,
                    extreme_movement_count=extremes,
                    trend_strength=0.1  # Neutral trend
                )
                
                # Verify risk level matches expected
                self.assertEqual(risk_level, expected,
                    f"Expected {expected} for volatility score {score} with {extremes} extreme movements")


    def test_analyze_transaction_anomalies(self):
        """Test transaction anomaly analysis"""
        # Set up mock return value
        self.calculator.db.execute_query.return_value = self.mock_transaction_data

        # Test with default parameters
        result = self.calculator.analyze_transaction_anomalies(
            lookback_days=self.lookback_days
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('anomaly_metrics', result)
        self.assertIn('risk_metrics', result)
        self.assertIn('anomalies', result)
        self.assertIn('clustering_analysis', result)
        self.assertIn('period', result)

        # Verify anomaly metrics
        metrics = result['anomaly_metrics']
        self.assertIn('total_transactions', metrics)
        self.assertIn('amount_anomalies', metrics)
        self.assertIn('pattern_anomalies', metrics)
        self.assertIn('anomaly_rate', metrics)

        # Verify risk metrics
        risk = result['risk_metrics']
        self.assertIn('risk_score', risk)
        self.assertIn('risk_level', risk)
        self.assertIn('risk_factors', risk)

        # Verify anomaly details
        anomalies = result['anomalies']
        self.assertIn('amount_based', anomalies)
        self.assertIn('pattern_based', anomalies)

        # Verify clustering analysis
        clustering = result['clustering_analysis']
        self.assertIn('cluster_count', clustering)
        self.assertIn('max_cluster_size', clustering)
        self.assertIn('cluster_details', clustering)

    def test_detect_pattern_anomalies(self):
        """Test pattern anomaly detection"""
        # Test data with known pattern anomalies
        test_data = self.mock_transaction_data.copy()
        
        # Add some extreme pattern anomalies
        test_data.iloc[10:15, test_data.columns.get_loc('line_count')] *= 5
        test_data.iloc[20:25, test_data.columns.get_loc('account_count')] *= 4

        # Detect anomalies
        pattern_anomalies = self.calculator._detect_pattern_anomalies(
            test_data,
            threshold_stddev=3.0
        )

        # Verify anomaly detection
        self.assertGreater(len(pattern_anomalies), 0)
        self.assertLessEqual(
            len(pattern_anomalies), 
            len(test_data)
        )

    def test_analyze_anomaly_clustering(self):
        """Test anomaly clustering analysis"""
        # Create test anomalies with known clustering
        test_dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        test_anomalies = pd.DataFrame({
            'journal_id': range(1, 11),
            'journal_date': [
                # First cluster (3 anomalies)
                test_dates[0], test_dates[1], test_dates[2],
                # Second cluster (2 anomalies)
                test_dates[15], test_dates[16],
                # Isolated anomalies
                test_dates[20], test_dates[25], test_dates[28], 
                test_dates[29], test_dates[29]
            ],
            'type': ['AMOUNT'] * 10,
            'max_amount': [1000] * 10  # Add required columns for clustering
        }).sort_values('journal_date')

        # Analyze clustering
        clustering = self.calculator._analyze_anomaly_clustering(test_anomalies)

        # Verify clustering results using more flexible assertions
        self.assertGreater(clustering['cluster_count'], 0)
        self.assertGreaterEqual(clustering['max_cluster_size'], 3)  # At least first cluster size
        self.assertGreater(len(clustering['cluster_details']), 0)
        
        # Verify cluster details
        for cluster in clustering['cluster_details']:
            self.assertIn('start_date', cluster)
            self.assertIn('end_date', cluster)
            self.assertIn('size', cluster)
            self.assertGreater(cluster['size'], 0)

    def test_anomaly_risk_metrics(self):
        """Test anomaly risk metric calculations"""
        test_metrics = {
            'total_transactions': 100,
            'amount_anomalies': 5,
            'pattern_anomalies': 3,
            'total_anomalies': 8,
            'anomaly_rate': 8.0
        }

        test_clustering = {
            'cluster_count': 2,
            'max_cluster_size': 3,
            'avg_cluster_size': 2.5
        }

        risk_metrics = self.calculator._calculate_anomaly_risk_metrics(
            test_metrics,
            test_clustering
        )

        # Verify risk metric structure and values
        self.assertIn('risk_score', risk_metrics)
        self.assertIn('risk_level', risk_metrics)
        self.assertIn('risk_factors', risk_metrics)
        
        self.assertGreaterEqual(risk_metrics['risk_score'], 0)
        self.assertLessEqual(risk_metrics['risk_score'], 100)
        self.assertIn(risk_metrics['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])

    
    def test_calculate_combined_risk_profile(self):
        """Test combined risk profile calculation"""
        # Set up mock return values for component analyses
        self.calculator.analyze_account_volatility = Mock(return_value={
            'status': 'ANALYZED',
            'stability_metrics': {
                'volatility_score': 45.0
            },
            'risk_level': 'MEDIUM'
        })

        self.calculator.analyze_transaction_anomalies = Mock(return_value={
            'status': 'ANALYZED',
            'risk_metrics': {
                'risk_score': 35.0,
                'risk_level': 'MEDIUM'
            }
        })

        # Override internal method to return known values
        self.calculator._analyze_risk_trends = Mock(return_value={
            'status': 'ANALYZED',
            'trend_risk_score': 25.0,
            'trend_direction': 'STABLE'
        })

        # Calculate combined risk profile
        result = self.calculator.calculate_combined_risk_profile(
            account_id=1001,
            lookback_days=self.lookback_days
        )

        # Verify structure
        self.assertIn('status', result)
        self.assertIn('composite_risk', result)
        self.assertIn('component_analysis', result)
        self.assertIn('recommendations', result)
        self.assertIn('period', result)

        # Verify composite risk
        composite_risk = result['composite_risk']
        self.assertIn('risk_score', composite_risk)
        self.assertIn('risk_level', composite_risk)
        self.assertIn('confidence_level', composite_risk)

        # Verify score bounds
        self.assertGreaterEqual(composite_risk['risk_score'], 0)
        self.assertLessEqual(composite_risk['risk_score'], 100)

        # Verify component analysis
        components = result['component_analysis']
        self.assertIn('volatility', components)
        self.assertIn('anomalies', components)
        self.assertIn('trends', components)

        # Verify recommendations
        self.assertIsInstance(result['recommendations'], list)
        self.assertGreater(len(result['recommendations']), 0)

    def test_calculate_composite_risk_score(self):
        """Test composite risk score calculation"""
        test_cases = [
            # (volatility, anomaly, trend, expected_level)
            (20.0, 20.0, 20.0, 'LOW'),
            (50.0, 50.0, 50.0, 'MEDIUM'),
            (80.0, 80.0, 80.0, 'HIGH')
        ]

        for vol_score, anom_score, trend_score, expected in test_cases:
            with self.subTest(vol_score=vol_score, anom_score=anom_score):
                score = self.calculator._calculate_composite_risk_score(
                    volatility_score=vol_score,
                    anomaly_score=anom_score,
                    trend_score=trend_score
                )
                
                risk_level = self.calculator._determine_risk_level(score)
                self.assertEqual(risk_level, expected)

    def test_calculate_confidence_level(self):
        """Test confidence level calculation"""
        # Test with all analyses having data
        high_confidence = self.calculator._calculate_confidence_level([
            {'status': 'ANALYZED'},
            {'status': 'ANALYZED'},
            {'status': 'ANALYZED'}
        ])
        self.assertEqual(high_confidence, 'HIGH')

        # Test with majority having data
        medium_confidence = self.calculator._calculate_confidence_level([
            {'status': 'ANALYZED'},
            {'status': 'ANALYZED'},
            {'status': 'NO_DATA'}
        ])
        self.assertEqual(medium_confidence, 'MEDIUM')

        # Test with minority having data
        low_confidence = self.calculator._calculate_confidence_level([
            {'status': 'NO_DATA'},
            {'status': 'NO_DATA'},
            {'status': 'ANALYZED'}
        ])
        self.assertEqual(low_confidence, 'LOW')  # Only one out of three has data

    def test_generate_risk_recommendations(self):
        """Test risk recommendation generation"""
        # Test high risk case
        high_risk_recs = self.calculator._generate_risk_recommendations(
            volatility_analysis={'risk_level': 'HIGH'},
            anomaly_analysis={'risk_metrics': {'risk_level': 'HIGH'}},
            trend_analysis={'trend_direction': 'DETERIORATING'},
            combined_score=80.0
        )
        self.assertGreater(len(high_risk_recs), 0)
        self.assertTrue(any('immediate' in rec.lower() for rec in high_risk_recs))

        # Test medium risk case
        medium_risk_recs = self.calculator._generate_risk_recommendations(
            volatility_analysis={'risk_level': 'MEDIUM'},
            anomaly_analysis={'risk_metrics': {'risk_level': 'MEDIUM'}},
            trend_analysis={'trend_direction': 'STABLE'},
            combined_score=50.0
        )
        self.assertGreater(len(medium_risk_recs), 0)
        self.assertTrue(any('monitor' in rec.lower() for rec in medium_risk_recs))

        # Test low risk case
        low_risk_recs = self.calculator._generate_risk_recommendations(
            volatility_analysis={'risk_level': 'LOW'},
            anomaly_analysis={'risk_metrics': {'risk_level': 'LOW'}},
            trend_analysis={'trend_direction': 'IMPROVING'},
            combined_score=20.0
        )
        self.assertEqual(len(low_risk_recs), 1)
        self.assertTrue('maintain' in low_risk_recs[0].lower())

if __name__ == '__main__':
    unittest.main()