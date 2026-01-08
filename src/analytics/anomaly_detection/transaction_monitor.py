# src/analytics/anomaly_detection/transaction_monitor.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import joblib
import json
import os

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class TransactionMonitor:
    """
    Monitors and detects anomalies in financial transactions.
    Integrates with existing GL and AP/AR data structure.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize monitor with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.model_utils = ModelUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.isolation_forest = None
        self.lof_model = None
        self.envelope_model = None
        self.scaler = None
        self.feature_columns = None
        self.training_period = None
        self.model_metrics = {}
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'amount': 3.0,  # Standard deviations for amount-based anomalies
            'volume': 2.5,  # Standard deviations for volume-based anomalies
            'pattern': 0.9,  # Isolation Forest contamination parameter
            'lof_neighbors': 20,  # Number of neighbors for LOF
            'confidence': 0.99  # Confidence level for Elliptic Envelope
        }
        
    def prepare_monitoring_data(self,
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """
        Prepare data for transaction monitoring
        
        Args:
            start_date: Start of monitoring period
            end_date: End of monitoring period
            
        Returns:
            DataFrame with features for anomaly detection
        """
        try:
            # Get required data
            gl_data = self._get_gl_data(start_date, end_date)
            ap_data = self._get_ap_data(start_date, end_date)
            journal_data = self._get_journal_data(start_date, end_date)
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. Transaction amount features
            amount_features = self._create_amount_features(gl_data)
            features_df = pd.concat([features_df, amount_features], axis=1)
            
            # 2. Transaction volume features
            volume_features = self._create_volume_features(gl_data, journal_data)
            features_df = pd.concat([features_df, volume_features], axis=1)
            
            # 3. Payment pattern features
            payment_features = self._create_payment_features(ap_data)
            features_df = pd.concat([features_df, payment_features], axis=1)
            
            # 4. Account interaction features
            interaction_features = self._create_interaction_features(gl_data)
            features_df = pd.concat([features_df, interaction_features], axis=1)
            
            # 5. Timing features
            timing_features = self._create_timing_features(journal_data)
            features_df = pd.concat([features_df, timing_features], axis=1)
            
            # Store feature columns for later use
            self.feature_columns = features_df.columns.tolist()
            self.training_period = (start_date, end_date)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing monitoring data: {str(e)}")
            raise

    def _get_gl_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get GL transaction data"""
        try:
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            
            # Filter for date range
            filtered_journals = gl_journals[
                (gl_journals['journal_date'].between(start_date, end_date)) &
                (gl_journals['status'] == 'POSTED')
            ]
            
            # Get transaction details
            gl_data = gl_lines[
                gl_lines['journal_id'].isin(filtered_journals['journal_id'])
            ].merge(
                filtered_journals[['journal_id', 'journal_date', 'journal_number']],
                on='journal_id'
            )
            
            return gl_data
            
        except Exception as e:
            self.logger.error(f"Error getting GL data: {str(e)}")
            raise

    def _get_ap_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get AP transaction data"""
        try:
            ap_invoices = self.data_loader.data['ap_invoices']
            ap_payments = self.data_loader.data['ap_payments']
            
            # Filter invoices
            invoices = ap_invoices[
                ap_invoices['invoice_date'].between(start_date, end_date)
            ]
            
            # Filter payments
            payments = ap_payments[
                ap_payments['payment_date'].between(start_date, end_date)
            ]
            
            # Combine with payment information
            ap_data = pd.merge(
                invoices,
                payments[['invoice_id', 'payment_date', 'payment_method', 'amount']],
                on='invoice_id',
                how='left'
            )
            
            return ap_data
            
        except Exception as e:
            self.logger.error(f"Error getting AP data: {str(e)}")
            raise

    def _get_journal_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get journal entry metadata"""
        try:
            gl_journals = self.data_loader.data['gl_journals']
            
            # Filter and enrich journal data
            journal_data = gl_journals[
                (gl_journals['journal_date'].between(start_date, end_date)) &
                (gl_journals['status'] == 'POSTED')
            ].copy()
            
            # Add timing information
            journal_data['hour'] = journal_data['journal_date'].dt.hour
            journal_data['day_of_week'] = journal_data['journal_date'].dt.dayofweek
            journal_data['day_of_month'] = journal_data['journal_date'].dt.day
            
            return journal_data
            
        except Exception as e:
            self.logger.error(f"Error getting journal data: {str(e)}")
            raise

    

    def _create_amount_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on transaction amounts"""
        try:
            # Group by journal
            amount_features = gl_data.groupby('journal_id').agg({
                'debit_amount': ['sum', 'mean', 'std', 'max', 'count'],
                'credit_amount': ['sum', 'mean', 'std', 'max', 'count']
            })
            
            # Flatten column names
            amount_features.columns = [
                'total_debits',
                'avg_debit',
                'std_debit',
                'max_debit',
                'debit_lines',
                'total_credits',
                'avg_credit',
                'std_credit',
                'max_credit',
                'credit_lines'
            ]
            
            # Calculate additional metrics
            amount_features['net_amount'] = (
                amount_features['total_debits'] - amount_features['total_credits']
            )
            
            amount_features['amount_imbalance'] = abs(
                amount_features['total_debits'] - amount_features['total_credits']
            )
            
            amount_features['avg_line_amount'] = (
                (amount_features['total_debits'] + amount_features['total_credits']) /
                (amount_features['debit_lines'] + amount_features['credit_lines'])
            ).fillna(0)
            
            # Calculate relative metrics
            mean_transaction = amount_features['avg_line_amount'].mean()
            std_transaction = amount_features['avg_line_amount'].std()
            
            amount_features['amount_zscore'] = (
                (amount_features['avg_line_amount'] - mean_transaction) /
                std_transaction
            ).fillna(0)
            
            return amount_features
            
        except Exception as e:
            self.logger.error(f"Error creating amount features: {str(e)}")
            raise

    def _create_volume_features(self, 
                              gl_data: pd.DataFrame,
                              journal_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on transaction volumes"""
        try:
            # Group by date
            daily_volumes = gl_data.groupby('journal_date').agg({
                'journal_id': 'nunique',  # Number of journals
                'line_id': 'count',      # Number of lines
                'account_id': 'nunique'  # Number of accounts
            })
            
            daily_volumes.columns = [
                'journal_count',
                'line_count',
                'unique_accounts'
            ]
            
            # Calculate volume metrics
            volume_features = pd.DataFrame()
            
            # Transaction intensity
            volume_features['transaction_intensity'] = (
                daily_volumes['line_count'] / daily_volumes['journal_count']
            ).fillna(0)
            
            # Account interaction density
            volume_features['account_density'] = (
                daily_volumes['unique_accounts'] / daily_volumes['line_count']
            ).fillna(0)
            
            # Calculate rolling statistics
            for window in [7, 30]:
                volume_features[f'rolling_{window}d_intensity'] = (
                    volume_features['transaction_intensity']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                volume_features[f'rolling_{window}d_density'] = (
                    volume_features['account_density']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            
            # Add volume change rates
            volume_features['volume_change'] = daily_volumes['line_count'].pct_change()
            volume_features['journal_change'] = daily_volumes['journal_count'].pct_change()
            
            return volume_features
            
        except Exception as e:
            self.logger.error(f"Error creating volume features: {str(e)}")
            raise

    def _create_payment_features(self, ap_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on payment patterns"""
        try:
            # Group by payment date
            payment_features = pd.DataFrame()
            
            # Calculate payment timing metrics
            payment_features['payment_delay'] = (
                ap_data['payment_date'] - ap_data['due_date']
            ).dt.days
            
            payment_features['early_payment'] = payment_features['payment_delay'] < 0
            payment_features['late_payment'] = payment_features['payment_delay'] > 0
            
            # Calculate payment amount metrics
            ap_data['payment_ratio'] = ap_data['amount_y'] / ap_data['amount_x']
            
            payment_features['overpayment'] = ap_data['payment_ratio'] > 1
            payment_features['underpayment'] = ap_data['payment_ratio'] < 1
            
            # Group payments by method
            method_counts = pd.get_dummies(
                ap_data['payment_method'],
                prefix='payment_method'
            )
            payment_features = pd.concat([payment_features, method_counts], axis=1)
            
            # Calculate payment patterns
            payment_features['payment_irregularity'] = abs(
                payment_features['payment_delay'].std()
            )
            
            return payment_features
            
        except Exception as e:
            self.logger.error(f"Error creating payment features: {str(e)}")
            raise

    def _create_interaction_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on account interactions"""
        try:
            # Create account interaction matrix
            interactions = pd.DataFrame()
            
            # Group by journal to get account pairs
            journal_accounts = gl_data.groupby('journal_id')['account_id'].agg(list)
            
            # Calculate account co-occurrence
            account_pairs = []
            for accounts in journal_accounts:
                pairs = [(min(a1, a2), max(a1, a2)) 
                        for i, a1 in enumerate(accounts) 
                        for a2 in accounts[i+1:]]
                account_pairs.extend(pairs)
            
            # Convert to DataFrame
            pair_df = pd.DataFrame(account_pairs, columns=['account1', 'account2'])
            pair_counts = pair_df.groupby(['account1', 'account2']).size().reset_index(name='frequency')
            
            # Calculate interaction metrics
            interactions['unique_interactions'] = len(pair_counts)
            interactions['avg_interaction_freq'] = pair_counts['frequency'].mean()
            interactions['max_interaction_freq'] = pair_counts['frequency'].max()
            
            # Calculate network metrics
            interactions['network_density'] = (
                2 * len(pair_counts) /
                (len(gl_data['account_id'].unique()) * (len(gl_data['account_id'].unique()) - 1))
            )
            
            # Add temporal aspect
            daily_interactions = gl_data.groupby('journal_date').apply(
                lambda x: len(pd.DataFrame(
                    [(min(a1, a2), max(a1, a2)) 
                     for i, a1 in enumerate(x['account_id']) 
                     for a2 in x['account_id'][i+1:]]
                ).drop_duplicates())
            )
            
            interactions['daily_interaction_avg'] = daily_interactions.mean()
            interactions['daily_interaction_std'] = daily_interactions.std()
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            raise

    def _create_timing_features(self, journal_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on transaction timing"""
        try:
            # Initialize timing features
            timing_features = pd.DataFrame()
            
            # Hour-based patterns
            hour_counts = pd.get_dummies(journal_data['hour'], prefix='hour')
            timing_features = pd.concat([timing_features, hour_counts], axis=1)
            
            # Day of week patterns
            dow_counts = pd.get_dummies(journal_data['day_of_week'], prefix='day')
            timing_features = pd.concat([timing_features, dow_counts], axis=1)
            
            # Calculate timing metrics
            timing_features['nonbusiness_hour_ratio'] = (
                (journal_data['hour'] < 9) | (journal_data['hour'] > 17)
            ).mean()
            
            timing_features['weekend_ratio'] = (
                journal_data['day_of_week'].isin([5, 6])
            ).mean()
            
            # Time intervals between transactions
            journal_data = journal_data.sort_values('journal_date')
            time_diffs = journal_data['journal_date'].diff().dt.total_seconds() / 3600
            
            timing_features['avg_time_between'] = time_diffs.mean()
            timing_features['min_time_between'] = time_diffs.min()
            timing_features['max_time_between'] = time_diffs.max()
            
            # Add temporal density features
            for window in [4, 8, 24]:  # hours
                timing_features[f'density_{window}h'] = (
                    journal_data.set_index('journal_date')
                    .rolling(f'{window}H')
                    .journal_id.count()
                    .mean()
                )
            
            return timing_features
            
        except Exception as e:
            self.logger.error(f"Error creating timing features: {str(e)}")
            raise


    

    def train_detector(self,
                      monitoring_data: pd.DataFrame,
                      contamination: float = 0.1) -> Dict:
        """
        Train anomaly detection models
        
        Args:
            monitoring_data: DataFrame with features for anomaly detection
            contamination: Expected proportion of outliers
            
        Returns:
            Dict containing training results
        """
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(monitoring_data)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            if_scores = self.isolation_forest.fit_predict(X_scaled)
            
            # Train Local Outlier Factor
            self.lof_model = LocalOutlierFactor(
                n_neighbors=self.anomaly_thresholds['lof_neighbors'],
                contamination=contamination
            )
            lof_scores = self.lof_model.fit_predict(X_scaled)
            
            # Train Robust Covariance (Elliptic Envelope)
            self.envelope_model = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=0.9
            )
            envelope_scores = self.envelope_model.fit_predict(X_scaled)
            
            # Combine model predictions
            combined_scores = np.vstack([if_scores, lof_scores, envelope_scores])
            ensemble_predictions = np.mean(combined_scores, axis=0)
            
            # Calculate metrics
            metrics = self._calculate_detection_metrics(
                monitoring_data,
                if_scores,
                lof_scores,
                envelope_scores,
                ensemble_predictions
            )
            
            self.model_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training anomaly detector: {str(e)}")
            raise

    def _calculate_detection_metrics(self,
                                  data: pd.DataFrame,
                                  if_scores: np.ndarray,
                                  lof_scores: np.ndarray,
                                  envelope_scores: np.ndarray,
                                  ensemble_scores: np.ndarray) -> Dict:
        """Calculate anomaly detection metrics"""
        try:
            metrics = {
                'isolation_forest': {
                    'anomaly_count': np.sum(if_scores == -1),
                    'anomaly_rate': np.mean(if_scores == -1) * 100
                },
                'lof': {
                    'anomaly_count': np.sum(lof_scores == -1),
                    'anomaly_rate': np.mean(lof_scores == -1) * 100
                },
                'envelope': {
                    'anomaly_count': np.sum(envelope_scores == -1),
                    'anomaly_rate': np.mean(envelope_scores == -1) * 100
                },
                'ensemble': {
                    'anomaly_count': np.sum(ensemble_scores < -0.5),
                    'anomaly_rate': np.mean(ensemble_scores < -0.5) * 100
                }
            }
            
            # Add model agreement metrics
            agreement_matrix = np.vstack([
                if_scores == -1,
                lof_scores == -1,
                envelope_scores == -1
            ])
            
            metrics['model_agreement'] = {
                'full_agreement': np.mean(np.all(agreement_matrix, axis=0)) * 100,
                'partial_agreement': np.mean(np.sum(agreement_matrix, axis=0) >= 2) * 100,
                'disagreement': np.mean(np.sum(agreement_matrix, axis=0) == 1) * 100
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating detection metrics: {str(e)}")
            raise

    def detect_anomalies(self,
                        transaction_data: pd.DataFrame,
                        detection_date: Optional[datetime] = None) -> Dict:
        """
        Detect anomalies in transaction data
        
        Args:
            transaction_data: DataFrame with transaction data
            detection_date: Optional specific date for detection
            
        Returns:
            Dict containing detected anomalies
        """
        try:
            if self.isolation_forest is None:
                raise ValueError("Models have not been trained yet")
            
            # Prepare features
            features = self._prepare_detection_features(transaction_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from each model
            if_predictions = self.isolation_forest.predict(features_scaled)
            lof_predictions = self.lof_model.predict(features_scaled)
            envelope_predictions = self.envelope_model.predict(features_scaled)
            
            # Combine predictions
            combined_scores = np.vstack([
                if_predictions,
                lof_predictions,
                envelope_predictions
            ])
            ensemble_predictions = np.mean(combined_scores, axis=0)
            
            # Get anomaly scores
            anomaly_scores = self._calculate_anomaly_scores(
                features_scaled,
                if_predictions,
                ensemble_predictions
            )
            
            # Identify anomalies based on ensemble predictions
            anomalies = self._identify_anomalies(
                transaction_data,
                features,
                ensemble_predictions,
                anomaly_scores
            )
            
            return {
                'anomalies': anomalies,
                'detection_summary': {
                    'total_transactions': len(transaction_data),
                    'anomaly_count': len(anomalies),
                    'anomaly_rate': (len(anomalies) / len(transaction_data)) * 100,
                    'detection_date': detection_date or datetime.now(),
                    'model_metrics': self.model_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise

    def _calculate_anomaly_scores(self,
                                features_scaled: np.ndarray,
                                if_predictions: np.ndarray,
                                ensemble_predictions: np.ndarray) -> pd.DataFrame:
        """Calculate detailed anomaly scores"""
        try:
            # Get anomaly scores from Isolation Forest
            if_scores = self.isolation_forest.score_samples(features_scaled)
            
            # Calculate distance-based scores
            distances = np.sqrt(np.sum(features_scaled ** 2, axis=1))
            
            # Normalize scores to 0-1 range
            normalized_if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
            normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
            
            # Combine scores
            anomaly_scores = pd.DataFrame({
                'isolation_score': normalized_if_scores,
                'distance_score': normalized_distances,
                'ensemble_score': ensemble_predictions,
                'is_anomaly': if_predictions == -1
            })
            
            # Calculate composite score
            anomaly_scores['composite_score'] = (
                0.4 * anomaly_scores['isolation_score'] +
                0.3 * anomaly_scores['distance_score'] +
                0.3 * anomaly_scores['ensemble_score']
            )
            
            return anomaly_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly scores: {str(e)}")
            raise

    def _identify_anomalies(self,
                          transaction_data: pd.DataFrame,
                          features: pd.DataFrame,
                          ensemble_predictions: np.ndarray,
                          anomaly_scores: pd.DataFrame) -> List[Dict]:
        """Identify and categorize anomalies"""
        try:
            anomalies = []
            
            # Combine data for analysis
            analysis_data = pd.concat([
                transaction_data.reset_index(drop=True),
                features.reset_index(drop=True),
                anomaly_scores
            ], axis=1)
            
            # Filter anomalous transactions
            anomalous_data = analysis_data[ensemble_predictions < -0.5]
            
            for _, row in anomalous_data.iterrows():
                anomaly_info = {
                    'transaction_id': row.get('journal_id') or row.get('payment_id'),
                    'transaction_date': row.get('journal_date') or row.get('payment_date'),
                    'anomaly_score': float(row['composite_score']),
                    'anomaly_factors': self._get_anomaly_factors(row),
                    'risk_level': self._determine_risk_level(row['composite_score']),
                    'details': self._extract_anomaly_details(row)
                }
                anomalies.append(anomaly_info)
            
            return sorted(
                anomalies,
                key=lambda x: x['anomaly_score'],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error identifying anomalies: {str(e)}")
            raise

    def _get_anomaly_factors(self, transaction: pd.Series) -> List[str]:
        """Determine factors contributing to anomaly"""
        factors = []
        
        # Amount-based factors
        if transaction.get('amount_zscore', 0) > self.anomaly_thresholds['amount']:
            factors.append('UNUSUAL_AMOUNT')
            
        if transaction.get('amount_imbalance', 0) > 0:
            factors.append('UNBALANCED_AMOUNT')
            
        # Volume-based factors
        if transaction.get('transaction_intensity', 0) > self.anomaly_thresholds['volume']:
            factors.append('HIGH_VOLUME')
            
        # Timing factors
        if transaction.get('nonbusiness_hour_ratio', 0) > 0.5:
            factors.append('UNUSUAL_TIMING')
            
        # Pattern factors
        if transaction.get('isolation_score', 0) < self.anomaly_thresholds['pattern']:
            factors.append('UNUSUAL_PATTERN')
        
        return factors
    

    

    def _determine_risk_level(self, anomaly_score: float) -> str:
        """Determine risk level based on anomaly score"""
        if anomaly_score >= 0.8:
            return 'CRITICAL'
        elif anomaly_score >= 0.6:
            return 'HIGH'
        elif anomaly_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _extract_anomaly_details(self, transaction: pd.Series) -> Dict:
        """Extract detailed information about anomalous transaction"""
        try:
            details = {
                'transaction_type': 'JOURNAL' if 'journal_id' in transaction else 'PAYMENT',
                'amount_metrics': {
                    'total_amount': float(transaction.get('total_debits', 0)),
                    'average_amount': float(transaction.get('avg_debit', 0)),
                    'amount_deviation': float(transaction.get('amount_zscore', 0))
                },
                'volume_metrics': {
                    'line_count': int(transaction.get('line_count', 0)),
                    'account_count': int(transaction.get('unique_accounts', 0)),
                    'transaction_intensity': float(transaction.get('transaction_intensity', 0))
                },
                'timing_metrics': {
                    'hour_of_day': int(transaction.get('hour', 0)),
                    'day_of_week': int(transaction.get('day_of_week', 0)),
                    'is_business_hours': not bool(transaction.get('nonbusiness_hour_ratio', 0))
                }
            }
            
            # Add model-specific scores
            details['model_scores'] = {
                'isolation_forest': float(transaction.get('isolation_score', 0)),
                'distance_based': float(transaction.get('distance_score', 0)),
                'ensemble': float(transaction.get('ensemble_score', 0))
            }
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error extracting anomaly details: {str(e)}")
            raise

    def analyze_detection_patterns(self, 
                                 start_date: datetime,
                                 end_date: datetime) -> Dict:
        """
        Analyze patterns in detected anomalies
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dict containing anomaly pattern analysis
        """
        try:
            # Get transaction data
            transaction_data = self._get_gl_data(start_date, end_date)
            ap_data = self._get_ap_data(start_date, end_date)
            
            # Detect anomalies
            detection_results = self.detect_anomalies(
                pd.concat([transaction_data, ap_data], axis=0)
            )
            
            anomalies = pd.DataFrame(detection_results['anomalies'])
            
            if anomalies.empty:
                return {
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'status': 'NO_ANOMALIES'
                }
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(anomalies)
            
            # Analyze factor patterns
            factor_patterns = self._analyze_factor_patterns(anomalies)
            
            # Analyze risk distribution
            risk_distribution = self._analyze_risk_distribution(anomalies)
            
            return {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'status': 'ANALYZED',
                'temporal_patterns': temporal_patterns,
                'factor_patterns': factor_patterns,
                'risk_distribution': risk_distribution,
                'summary_metrics': {
                    'total_anomalies': len(anomalies),
                    'high_risk_rate': (
                        len(anomalies[anomalies['risk_level'].isin(['HIGH', 'CRITICAL'])]) /
                        len(anomalies) * 100
                    ),
                    'avg_anomaly_score': float(anomalies['anomaly_score'].mean())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing detection patterns: {str(e)}")
            raise

    def _analyze_temporal_patterns(self, anomalies: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in anomalies"""
        try:
            # Convert transaction_date to datetime if string
            anomalies['transaction_date'] = pd.to_datetime(anomalies['transaction_date'])
            
            # Daily patterns
            daily_counts = anomalies.groupby(
                anomalies['transaction_date'].dt.date
            ).size()
            
            # Hour of day patterns
            hour_counts = anomalies.groupby(
                anomalies['transaction_date'].dt.hour
            ).size()
            
            # Day of week patterns
            dow_counts = anomalies.groupby(
                anomalies['transaction_date'].dt.dayofweek
            ).size()
            
            return {
                'daily_distribution': daily_counts.to_dict(),
                'hourly_distribution': hour_counts.to_dict(),
                'weekday_distribution': dow_counts.to_dict(),
                'temporal_metrics': {
                    'avg_daily_anomalies': float(daily_counts.mean()),
                    'max_daily_anomalies': int(daily_counts.max()),
                    'business_hours_rate': float(
                        hour_counts[9:17].sum() / hour_counts.sum() * 100
                    ),
                    'weekend_rate': float(
                        dow_counts[[5,6]].sum() / dow_counts.sum() * 100
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {str(e)}")
            raise

    def _analyze_factor_patterns(self, anomalies: pd.DataFrame) -> Dict:
        """Analyze patterns in anomaly factors"""
        try:
            # Extract factors from list column
            all_factors = [
                factor
                for factors in anomalies['anomaly_factors']
                for factor in factors
            ]
            
            # Calculate factor frequencies
            factor_counts = pd.Series(all_factors).value_counts()
            
            # Calculate co-occurrence matrix
            factor_pairs = []
            for factors in anomalies['anomaly_factors']:
                pairs = [(f1, f2) for i, f1 in enumerate(factors) 
                        for f2 in factors[i+1:]]
                factor_pairs.extend(pairs)
                
            cooccurrence = pd.Series(factor_pairs).value_counts()
            
            return {
                'factor_frequencies': factor_counts.to_dict(),
                'factor_cooccurrence': {
                    f'{f1}-{f2}': count
                    for (f1, f2), count in cooccurrence.items()
                },
                'factor_metrics': {
                    'avg_factors_per_anomaly': float(
                        len(all_factors) / len(anomalies)
                    ),
                    'most_common_factor': factor_counts.index[0],
                    'most_common_pair': f'{cooccurrence.index[0][0]}-{cooccurrence.index[0][1]}'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing factor patterns: {str(e)}")
            raise

    def _analyze_risk_distribution(self, anomalies: pd.DataFrame) -> Dict:
        """Analyze distribution of risk levels"""
        try:
            # Calculate risk level distribution
            risk_counts = anomalies['risk_level'].value_counts()
            
            # Calculate score distribution metrics
            score_quartiles = anomalies['anomaly_score'].quantile([0.25, 0.5, 0.75])
            
            return {
                'risk_distribution': risk_counts.to_dict(),
                'score_distribution': {
                    'min_score': float(anomalies['anomaly_score'].min()),
                    'max_score': float(anomalies['anomaly_score'].max()),
                    'avg_score': float(anomalies['anomaly_score'].mean()),
                    'quartiles': {
                        '25th': float(score_quartiles[0.25]),
                        '50th': float(score_quartiles[0.5]),
                        '75th': float(score_quartiles[0.75])
                    }
                },
                'risk_metrics': {
                    'critical_rate': float(
                        (anomalies['risk_level'] == 'CRITICAL').mean() * 100
                    ),
                    'high_risk_rate': float(
                        (anomalies['risk_level'] == 'HIGH').mean() * 100
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk distribution: {str(e)}")
            raise

    def save_detector(self, model_dir: str) -> None:
        """Save trained detector models"""
        try:
            if self.isolation_forest is None:
                raise ValueError("Models have not been trained yet")
                
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            joblib.dump(self.isolation_forest, 
                       os.path.join(model_dir, 'isolation_forest.joblib'))
            joblib.dump(self.lof_model,
                       os.path.join(model_dir, 'lof_model.joblib'))
            joblib.dump(self.envelope_model,
                       os.path.join(model_dir, 'envelope_model.joblib'))
            joblib.dump(self.scaler,
                       os.path.join(model_dir, 'scaler.joblib'))
            
            # Save model info
            model_info = {
                'feature_columns': self.feature_columns,
                'training_period': self.training_period,
                'model_metrics': self.model_metrics,
                'anomaly_thresholds': self.anomaly_thresholds,
                'save_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, default=str)
                
            self.logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def load_detector(self, model_dir: str) -> None:
        """Load saved detector models"""
        try:
            # Load models
            self.isolation_forest = joblib.load(
                os.path.join(model_dir, 'isolation_forest.joblib')
            )
            self.lof_model = joblib.load(
                os.path.join(model_dir, 'lof_model.joblib')
            )
            self.envelope_model = joblib.load(
                os.path.join(model_dir, 'envelope_model.joblib')
            )
            self.scaler = joblib.load(
                os.path.join(model_dir, 'scaler.joblib')
            )
            
            # Load model info
            with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
                model_info = json.load(f)
                
            self.feature_columns = model_info['feature_columns']
            self.training_period = tuple(
                datetime.strptime(d, '%Y-%m-%d')
                for d in model_info['training_period']
            )
            self.model_metrics = model_info['model_metrics']
            self.anomaly_thresholds = model_info['anomaly_thresholds']
            
            self.logger.info(f"Models loaded from {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    