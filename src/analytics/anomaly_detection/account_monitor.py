# src/analytics/anomaly_detection/account_monitor.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import joblib
import json
import os

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader
#from src.analytics.financial_analysis.account_metrics import AccountMetricsCalculator
from src.feature_engineering.financial_metrics.account_metrics import AccountMetricsCalculator

class AccountMonitor:
    """
    Monitors and detects anomalies in GL account activity.
    Integrates with existing GL data structure and account metrics.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize monitor with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.account_metrics = AccountMetricsCalculator(data_loader)
        self.model_utils = ModelUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.isolation_forest = None
        self.envelope_model = None
        self.lof_model = None
        self.scaler = None
        self.feature_columns = None
        self.training_period = None
        self.model_metrics = {}
        
        # Monitoring parameters
        self.monitor_params = {
            'min_history': 30,       # Minimum days of history required
            'outlier_threshold': 3.0, # Standard deviations for outlier detection
            'time_window': 90,       # Days for rolling window analysis
            'min_transactions': 5,    # Minimum transactions for pattern analysis
            'confidence_level': 0.95  # Confidence level for anomaly detection
        }
        
    def prepare_monitoring_data(self,
                              start_date: datetime,
                              end_date: datetime,
                              account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare data for account monitoring
        
        Args:
            start_date: Start of monitoring period
            end_date: End of monitoring period
            account_types: Optional list of account types to monitor
            
        Returns:
            DataFrame with features for monitoring
        """
        try:
            # Get required data
            gl_data = self._get_gl_data(start_date, end_date, account_types)
            balance_data = self._get_balance_data(start_date, end_date, account_types)
            account_data = self._get_account_data(account_types)
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. Transaction features
            transaction_features = self._create_transaction_features(gl_data)
            features_df = pd.concat([features_df, transaction_features], axis=1)
            
            # 2. Balance features
            balance_features = self._create_balance_features(balance_data)
            features_df = pd.concat([features_df, balance_features], axis=1)
            
            # 3. Account features
            account_features = self._create_account_features(account_data)
            features_df = pd.concat([features_df, account_features], axis=1)
            
            # 4. Activity pattern features
            pattern_features = self._create_pattern_features(gl_data)
            features_df = pd.concat([features_df, pattern_features], axis=1)
            
            # Store feature columns for later use
            self.feature_columns = features_df.columns.tolist()
            self.training_period = (start_date, end_date)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing monitoring data: {str(e)}")
            raise

    def _get_gl_data(self,
                     start_date: datetime,
                     end_date: datetime,
                     account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get GL transaction data"""
        try:
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_coa = self.data_loader.data['gl_coa']
            
            # Filter accounts by type if specified
            if account_types:
                account_ids = gl_coa[
                    gl_coa['account_type'].isin(account_types)
                ]['account_id'].tolist()
            else:
                account_ids = gl_coa['account_id'].tolist()
            
            # Filter journal entries
            filtered_journals = gl_journals[
                (gl_journals['journal_date'].between(start_date, end_date)) &
                (gl_journals['status'] == 'POSTED')
            ]
            
            # Get transaction details
            gl_data = gl_lines[
                (gl_lines['account_id'].isin(account_ids)) &
                (gl_lines['journal_id'].isin(filtered_journals['journal_id']))
            ].merge(
                filtered_journals[['journal_id', 'journal_date']],
                on='journal_id'
            )
            
            return gl_data
            
        except Exception as e:
            self.logger.error(f"Error getting GL data: {str(e)}")
            raise

    def _get_balance_data(self,
                         start_date: datetime,
                         end_date: datetime,
                         account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get account balance data"""
        try:
            # Initialize daily balance history
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            accounts = self._get_account_data(account_types)
            
            balance_history = pd.DataFrame()
            
            # Calculate daily balances for each account
            for account_id in accounts['account_id']:
                daily_balances = pd.DataFrame(index=date_range)
                daily_balances['account_id'] = account_id
                
                for date in date_range:
                    balance = self.account_metrics.calculate_account_balance(
                        account_id,
                        date
                    )
                    daily_balances.loc[date, 'balance'] = balance['balance']
                    daily_balances.loc[date, 'total_debits'] = balance['total_debits']
                    daily_balances.loc[date, 'total_credits'] = balance['total_credits']
                
                balance_history = pd.concat([balance_history, daily_balances])
            
            return balance_history
            
        except Exception as e:
            self.logger.error(f"Error getting balance data: {str(e)}")
            raise

    def _get_account_data(self,
                         account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get account master data"""
        try:
            gl_coa = self.data_loader.data['gl_coa']
            
            # Filter by account types if specified
            if account_types:
                account_data = gl_coa[gl_coa['account_type'].isin(account_types)].copy()
            else:
                account_data = gl_coa.copy()
            
            # Filter active accounts
            account_data = account_data[account_data['is_active'] == 'Y']
            
            return account_data
            
        except Exception as e:
            self.logger.error(f"Error getting account data: {str(e)}")
            raise


    

    def _create_transaction_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on GL transactions"""
        try:
            # Group by account and date
            transaction_features = gl_data.groupby(['account_id', 'journal_date']).agg({
                'journal_id': 'nunique',  # Number of journals
                'line_id': 'count',      # Number of lines
                'debit_amount': ['sum', 'mean', 'std', 'max'],
                'credit_amount': ['sum', 'mean', 'std', 'max']
            })
            
            # Flatten column names
            transaction_features.columns = [
                'journal_count',
                'line_count',
                'total_debits',
                'avg_debit',
                'std_debit',
                'max_debit',
                'total_credits',
                'avg_credit',
                'std_credit',
                'max_credit'
            ]
            
            # Calculate additional metrics
            transaction_features['net_movement'] = (
                transaction_features['total_debits'] -
                transaction_features['total_credits']
            )
            
            transaction_features['transaction_volume'] = (
                transaction_features['total_debits'] +
                transaction_features['total_credits']
            )
            
            transaction_features['average_transaction_size'] = (
                transaction_features['transaction_volume'] /
                transaction_features['line_count']
            ).fillna(0)
            
            # Add rolling metrics
            for window in [7, 30, 90]:
                transaction_features[f'rolling_{window}d_volume'] = (
                    transaction_features.groupby(level=0)['transaction_volume']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                transaction_features[f'rolling_{window}d_net'] = (
                    transaction_features.groupby(level=0)['net_movement']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                transaction_features[f'rolling_{window}d_volatility'] = (
                    transaction_features.groupby(level=0)['net_movement']
                    .rolling(window=window, min_periods=1)
                    .std()
                )
            
            return transaction_features
            
        except Exception as e:
            self.logger.error(f"Error creating transaction features: {str(e)}")
            raise

    def _create_balance_features(self, balance_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on account balances"""
        try:
            # Initialize balance features
            balance_features = pd.DataFrame()
            
            # Calculate balance metrics
            balance_features['current_balance'] = balance_data['balance']
            balance_features['balance_change'] = balance_data.groupby('account_id')['balance'].diff()
            
            # Rolling statistics
            for window in [7, 30, 90]:
                # Balance metrics
                balance_features[f'balance_{window}d_avg'] = (
                    balance_data.groupby('account_id')['balance']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                balance_features[f'balance_{window}d_std'] = (
                    balance_data.groupby('account_id')['balance']
                    .rolling(window=window, min_periods=1)
                    .std()
                )
                
                # Change metrics
                balance_features[f'change_{window}d_avg'] = (
                    balance_features['balance_change']
                    .groupby(level=0)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                balance_features[f'change_{window}d_std'] = (
                    balance_features['balance_change']
                    .groupby(level=0)
                    .rolling(window=window, min_periods=1)
                    .std()
                )
            
            # Calculate momentum indicators
            for period in [7, 30]:
                balance_features[f'momentum_{period}d'] = (
                    balance_features['current_balance'] -
                    balance_features[f'balance_{period}d_avg']
                )
            
            # Calculate volatility metrics
            balance_features['volatility'] = (
                balance_features['balance_30d_std'] /
                balance_features['balance_30d_avg']
            ).fillna(0)
            
            # Add trend direction
            balance_features['trend_direction'] = np.sign(
                balance_features['balance_30d_avg'].diff()
            )
            
            return balance_features
            
        except Exception as e:
            self.logger.error(f"Error creating balance features: {str(e)}")
            raise

    def _create_account_features(self, account_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on account characteristics"""
        try:
            # Initialize account features
            account_features = pd.DataFrame()
            
            # Account type features (one-hot encoding)
            type_dummies = pd.get_dummies(
                account_data['account_type'],
                prefix='account_type'
            )
            account_features = pd.concat([account_features, type_dummies], axis=1)
            
            # Account category features (one-hot encoding)
            category_dummies = pd.get_dummies(
                account_data['account_category'],
                prefix='category'
            )
            account_features = pd.concat([account_features, category_dummies], axis=1)
            
            # Account hierarchy features
            account_features['has_parent'] = account_data['parent_account_id'].notna().astype(int)
            account_features['account_level'] = self._calculate_account_level(account_data)
            
            # Account age features
            account_features['account_age_days'] = (
                datetime.now() - pd.to_datetime(account_data['created_date'])
            ).dt.days
            
            # Status features
            account_features['is_active'] = (account_data['is_active'] == 'Y').astype(int)
            
            # Last update features
            account_features['days_since_update'] = (
                datetime.now() - pd.to_datetime(account_data['last_updated_date'])
            ).dt.days
            
            return account_features
            
        except Exception as e:
            self.logger.error(f"Error creating account features: {str(e)}")
            raise

    def _calculate_account_level(self, account_data: pd.DataFrame) -> pd.Series:
        """Calculate hierarchical level of each account"""
        try:
            def get_level(account_id: int, visited: set = None) -> int:
                if visited is None:
                    visited = set()
                    
                if account_id in visited:
                    return 0  # Prevent infinite recursion
                    
                visited.add(account_id)
                parent_id = account_data.loc[
                    account_data['account_id'] == account_id,
                    'parent_account_id'
                ].iloc[0]
                
                if pd.isna(parent_id):
                    return 1
                else:
                    return 1 + get_level(parent_id, visited)
                    
            levels = account_data['account_id'].apply(get_level)
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating account levels: {str(e)}")
            raise

    def _create_pattern_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on transaction patterns"""
        try:
            # Group by account
            pattern_features = pd.DataFrame()
            
            # Transaction timing patterns
            pattern_features['day_of_week_entropy'] = gl_data.groupby('account_id').apply(
                lambda x: self._calculate_entropy(x['journal_date'].dt.dayofweek)
            )
            
            pattern_features['day_of_month_entropy'] = gl_data.groupby('account_id').apply(
                lambda x: self._calculate_entropy(x['journal_date'].dt.day)
            )
            
            pattern_features['hour_entropy'] = gl_data.groupby('account_id').apply(
                lambda x: self._calculate_entropy(x['journal_date'].dt.hour)
            )
            
            # Transaction amount patterns
            pattern_features['debit_entropy'] = gl_data.groupby('account_id').apply(
                lambda x: self._calculate_entropy(pd.qcut(x['debit_amount'], q=10, duplicates='drop'))
            )
            
            pattern_features['credit_entropy'] = gl_data.groupby('account_id').apply(
                lambda x: self._calculate_entropy(pd.qcut(x['credit_amount'], q=10, duplicates='drop'))
            )
            
            # Transaction regularity
            pattern_features['timing_regularity'] = gl_data.groupby('account_id').apply(
                self._calculate_timing_regularity
            )
            
            pattern_features['amount_regularity'] = gl_data.groupby('account_id').apply(
                self._calculate_amount_regularity
            )
            
            # Pattern complexity
            pattern_features['pattern_complexity'] = gl_data.groupby('account_id').apply(
                self._calculate_pattern_complexity
            )
            
            return pattern_features
            
        except Exception as e:
            self.logger.error(f"Error creating pattern features: {str(e)}")
            raise

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a series"""
        try:
            value_counts = series.value_counts(normalize=True)
            return float(-np.sum(value_counts * np.log2(value_counts)))
        except Exception:
            return 0.0

    def _calculate_timing_regularity(self, group: pd.DataFrame) -> float:
        """Calculate timing regularity score"""
        try:
            if len(group) < self.monitor_params['min_transactions']:
                return 0.0
                
            # Calculate intervals between transactions
            intervals = group['journal_date'].diff().dt.total_seconds()
            
            # Calculate coefficient of variation
            cv = intervals.std() / intervals.mean() if intervals.mean() != 0 else float('inf')
            
            # Convert to regularity score (0 to 1)
            regularity = 1 / (1 + cv)
            
            return float(regularity)
            
        except Exception as e:
            self.logger.error(f"Error calculating timing regularity: {str(e)}")
            return 0.0

    def _calculate_amount_regularity(self, group: pd.DataFrame) -> float:
        """Calculate amount regularity score"""
        try:
            if len(group) < self.monitor_params['min_transactions']:
                return 0.0
                
            # Calculate regularity for debits and credits
            debit_cv = group['debit_amount'].std() / group['debit_amount'].mean() if group['debit_amount'].mean() != 0 else float('inf')
            credit_cv = group['credit_amount'].std() / group['credit_amount'].mean() if group['credit_amount'].mean() != 0 else float('inf')
            
            # Convert to regularity scores
            debit_regularity = 1 / (1 + debit_cv)
            credit_regularity = 1 / (1 + credit_cv)
            
            # Return average regularity
            return float(np.mean([debit_regularity, credit_regularity]))
            
        except Exception as e:
            self.logger.error(f"Error calculating amount regularity: {str(e)}")
            return 0.0

    def _calculate_pattern_complexity(self, group: pd.DataFrame) -> float:
        """Calculate complexity of transaction patterns"""
        try:
            if len(group) < self.monitor_params['min_transactions']:
                return 0.0
                
            # Calculate various complexity metrics
            timing_complexity = self._calculate_entropy(group['journal_date'].dt.dayofweek)
            amount_complexity = len(pd.qcut(group['debit_amount'] + group['credit_amount'], q=4, duplicates='drop'))
            interval_complexity = self._calculate_entropy(group['journal_date'].diff().dt.days)
            
            # Combine into complexity score (0 to 1)
            complexity_score = np.mean([
                timing_complexity / 2.8,  # Normalize by max entropy for 7 days
                amount_complexity / 4.0,  # Normalize by number of quartiles
                interval_complexity / 5.0  # Normalize by typical max entropy
            ])
            
            return float(complexity_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern complexity: {str(e)}")
            return 0.0
        
    
    

    def train_monitor(self,
                     monitoring_data: pd.DataFrame,
                     contamination: float = 0.1) -> Dict:
        """
        Train anomaly detection models
        
        Args:
            monitoring_data: DataFrame with monitoring features
            contamination: Expected proportion of anomalies
            
        Returns:
            Dict containing training results
        """
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(monitoring_data)
            
            # Initialize and train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            if_scores = self.isolation_forest.fit_predict(X_scaled)
            
            # Initialize and train Robust Covariance
            self.envelope_model = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=0.9
            )
            envelope_scores = self.envelope_model.fit_predict(X_scaled)
            
            # Initialize and train Local Outlier Factor
            self.lof_model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination
            )
            lof_scores = self.lof_model.fit_predict(X_scaled)
            
            # Combine model predictions
            combined_scores = np.vstack([if_scores, envelope_scores, lof_scores])
            ensemble_predictions = np.mean(combined_scores, axis=0)
            
            # Calculate metrics
            metrics = self._calculate_detection_metrics(
                monitoring_data,
                if_scores,
                envelope_scores,
                lof_scores,
                ensemble_predictions
            )
            
            self.model_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training monitor: {str(e)}")
            raise

    def _calculate_detection_metrics(self,
                                  data: pd.DataFrame,
                                  if_scores: np.ndarray,
                                  envelope_scores: np.ndarray,
                                  lof_scores: np.ndarray,
                                  ensemble_scores: np.ndarray) -> Dict:
        """Calculate anomaly detection metrics"""
        try:
            metrics = {
                'isolation_forest': {
                    'anomaly_count': np.sum(if_scores == -1),
                    'anomaly_rate': np.mean(if_scores == -1) * 100
                },
                'elliptic_envelope': {
                    'anomaly_count': np.sum(envelope_scores == -1),
                    'anomaly_rate': np.mean(envelope_scores == -1) * 100
                },
                'lof': {
                    'anomaly_count': np.sum(lof_scores == -1),
                    'anomaly_rate': np.mean(lof_scores == -1) * 100
                },
                'ensemble': {
                    'anomaly_count': np.sum(ensemble_scores < -0.5),
                    'anomaly_rate': np.mean(ensemble_scores < -0.5) * 100
                }
            }
            
            # Calculate model agreement metrics
            agreement_matrix = np.vstack([
                if_scores == -1,
                envelope_scores == -1,
                lof_scores == -1
            ])
            
            metrics['model_agreement'] = {
                'full_agreement': np.mean(np.all(agreement_matrix, axis=0)) * 100,
                'partial_agreement': np.mean(np.sum(agreement_matrix, axis=0) >= 2) * 100,
                'disagreement': np.mean(np.sum(agreement_matrix, axis=0) == 1) * 100
            }
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(data)
            metrics['feature_importance'] = feature_importance
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating detection metrics: {str(e)}")
            raise

    def detect_anomalies(self,
                        account_data: pd.DataFrame,
                        detection_date: Optional[datetime] = None) -> Dict:
        """
        Detect anomalies in account activity
        
        Args:
            account_data: DataFrame with account data
            detection_date: Optional specific date for detection
            
        Returns:
            Dict containing detected anomalies
        """
        try:
            if self.isolation_forest is None:
                raise ValueError("Models have not been trained yet")
            
            # Prepare features for detection
            features = self._prepare_detection_features(account_data, detection_date)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from each model
            if_predictions = self.isolation_forest.predict(features_scaled)
            envelope_predictions = self.envelope_model.predict(features_scaled)
            lof_predictions = self.lof_model.predict(features_scaled)
            
            # Combine predictions
            combined_scores = np.vstack([
                if_predictions,
                envelope_predictions,
                lof_predictions
            ])
            ensemble_predictions = np.mean(combined_scores, axis=0)
            
            # Calculate anomaly scores
            anomaly_scores = self._calculate_anomaly_scores(
                features_scaled,
                ensemble_predictions
            )
            
            # Identify anomalies
            anomalies = self._identify_anomalies(
                account_data,
                features,
                ensemble_predictions,
                anomaly_scores
            )
            
            return {
                'anomalies': anomalies,
                'detection_summary': {
                    'total_accounts': len(account_data),
                    'anomaly_count': len(anomalies),
                    'anomaly_rate': (len(anomalies) / len(account_data)) * 100,
                    'detection_date': detection_date or datetime.now(),
                    'model_metrics': self.model_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise

    def _prepare_detection_features(self,
                                  account_data: pd.DataFrame,
                                  detection_date: Optional[datetime] = None) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        try:
            detection_date = detection_date or datetime.now()
            lookback_start = detection_date - timedelta(days=self.monitor_params['time_window'])
            
            # Get required data
            gl_data = self._get_gl_data(
                lookback_start,
                detection_date
            )
            
            balance_data = self._get_balance_data(
                lookback_start,
                detection_date
            )
            
            # Create features
            transaction_features = self._create_transaction_features(gl_data)
            balance_features = self._create_balance_features(balance_data)
            account_features = self._create_account_features(account_data)
            pattern_features = self._create_pattern_features(gl_data)
            
            # Combine features
            features = pd.concat([
                transaction_features.tail(1),
                balance_features.tail(1),
                account_features,
                pattern_features
            ], axis=1)
            
            # Ensure all feature columns are present in correct order
            features = features.reindex(columns=self.feature_columns, fill_value=0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing detection features: {str(e)}")
            raise

    def _calculate_anomaly_scores(self,
                                features_scaled: np.ndarray,
                                ensemble_predictions: np.ndarray) -> pd.DataFrame:
        """Calculate detailed anomaly scores"""
        try:
            # Get anomaly scores from each model
            if_scores = -self.isolation_forest.score_samples(features_scaled)
            lof_scores = -self.lof_model.score_samples(features_scaled)
            envelope_scores = -self.envelope_model.score_samples(features_scaled)
            
            # Normalize scores to 0-1 range
            def normalize_scores(scores: np.ndarray) -> np.ndarray:
                return (scores - scores.min()) / (scores.max() - scores.min())
            
            normalized_if = normalize_scores(if_scores)
            normalized_lof = normalize_scores(lof_scores)
            normalized_envelope = normalize_scores(envelope_scores)
            
            # Create score DataFrame
            anomaly_scores = pd.DataFrame({
                'isolation_score': normalized_if,
                'lof_score': normalized_lof,
                'envelope_score': normalized_envelope,
                'ensemble_score': ensemble_predictions,
                'is_anomaly': ensemble_predictions < -0.5
            })
            
            # Calculate composite score
            anomaly_scores['composite_score'] = (
                0.4 * anomaly_scores['isolation_score'] +
                0.3 * anomaly_scores['lof_score'] +
                0.3 * anomaly_scores['envelope_score']
            )
            
            return anomaly_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly scores: {str(e)}")
            raise

    def _calculate_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for anomaly detection"""
        try:
            feature_importance = {}
            
            # Use Isolation Forest feature importances as base
            importance_scores = self.isolation_forest.feature_importances_
            
            # Create dictionary of feature importance
            feature_importance = dict(zip(self.feature_columns, importance_scores))
            
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(),
                      key=lambda x: x[1],
                      reverse=True)
            )
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise

    def _identify_anomalies(self,
                            account_data: pd.DataFrame,
                            features: pd.DataFrame,
                            ensemble_predictions: np.ndarray,
                            anomaly_scores: pd.DataFrame) -> List[Dict]:
        """
        Identify and classify anomalies with detailed explanations
        
        Args:
            account_data: DataFrame with account information
            features: DataFrame with detection features
            ensemble_predictions: Array of ensemble model predictions
            anomaly_scores: DataFrame with anomaly scores
            
        Returns:
            List of dictionaries containing anomaly details
        """
        try:
            anomalies = []
            
            # Identify accounts with anomalies
            anomaly_indices = np.where(ensemble_predictions < -0.5)[0]
            
            for idx in anomaly_indices:
                account_info = account_data.iloc[idx]
                feature_values = features.iloc[idx]
                scores = anomaly_scores.iloc[idx]
                
                # Determine anomaly characteristics
                anomaly_types = self._classify_anomaly_types(feature_values)
                severity = self._calculate_anomaly_severity(scores)
                contributing_factors = self._identify_contributing_factors(
                    feature_values,
                    self.model_metrics['feature_importance']
                )
                
                # Generate detailed explanation
                explanation = self._generate_anomaly_explanation(
                    account_info,
                    anomaly_types,
                    contributing_factors,
                    severity
                )
                
                # Create anomaly record
                anomaly = {
                    'account_id': int(account_info['account_id']),
                    'account_number': account_info['account_number'],
                    'account_name': account_info['account_name'],
                    'detection_time': datetime.now(),
                    'anomaly_types': anomaly_types,
                    'severity': severity,
                    'scores': {
                        'isolation_forest': float(scores['isolation_score']),
                        'lof': float(scores['lof_score']),
                        'envelope': float(scores['envelope_score']),
                        'composite': float(scores['composite_score'])
                    },
                    'contributing_factors': contributing_factors,
                    'explanation': explanation,
                    'recommended_actions': self._get_recommended_actions(
                        anomaly_types,
                        severity,
                        account_info
                    )
                }
                
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error identifying anomalies: {str(e)}")
            raise

    def _classify_anomaly_types(self, feature_values: pd.Series) -> List[str]:
        """Classify the types of anomalies based on feature values"""
        try:
            anomaly_types = []
            
            # Volume anomalies
            if feature_values.get('transaction_volume', 0) > feature_values.get('rolling_30d_volume', 0) * 2:
                anomaly_types.append('HIGH_VOLUME')
            elif feature_values.get('transaction_volume', 0) < feature_values.get('rolling_30d_volume', 0) * 0.5:
                anomaly_types.append('LOW_VOLUME')
                
            # Balance anomalies
            if abs(feature_values.get('balance_change', 0)) > feature_values.get('balance_30d_std', 0) * 3:
                anomaly_types.append('UNUSUAL_BALANCE_CHANGE')
                
            # Pattern anomalies
            if feature_values.get('timing_regularity', 0) < 0.2:
                anomaly_types.append('IRREGULAR_TIMING')
            if feature_values.get('amount_regularity', 0) < 0.2:
                anomaly_types.append('IRREGULAR_AMOUNTS')
                
            # Trend anomalies
            if feature_values.get('momentum_30d', 0) > feature_values.get('balance_30d_std', 0) * 2:
                anomaly_types.append('UNUSUAL_POSITIVE_TREND')
            elif feature_values.get('momentum_30d', 0) < -feature_values.get('balance_30d_std', 0) * 2:
                anomaly_types.append('UNUSUAL_NEGATIVE_TREND')
                
            return anomaly_types or ['GENERAL_ANOMALY']
            
        except Exception as e:
            self.logger.error(f"Error classifying anomaly types: {str(e)}")
            return ['CLASSIFICATION_ERROR']

    def _calculate_anomaly_severity(self, scores: pd.Series) -> str:
        """Calculate anomaly severity based on scores"""
        try:
            composite_score = scores['composite_score']
            
            if composite_score > 0.8:
                return 'CRITICAL'
            elif composite_score > 0.6:
                return 'HIGH'
            elif composite_score > 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            self.logger.error(f"Error calculating anomaly severity: {str(e)}")
            return 'UNKNOWN'

    def _identify_contributing_factors(self,
                                    feature_values: pd.Series,
                                    feature_importance: Dict[str, float],
                                    top_n: int = 5) -> List[Dict]:
        """Identify factors contributing to the anomaly"""
        try:
            factors = []
            
            # Calculate normalized feature values
            normalized_values = {}
            for feature, value in feature_values.items():
                if feature in self.feature_columns:
                    normalized_values[feature] = abs(
                        (value - self.scaler.mean_[self.feature_columns.index(feature)]) /
                        self.scaler.scale_[self.feature_columns.index(feature)]
                    )
            
            # Combine with feature importance
            factor_scores = {
                feature: value * feature_importance.get(feature, 0)
                for feature, value in normalized_values.items()
            }
            
            # Get top contributing factors
            top_factors = sorted(
                factor_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            # Format factors with details
            for feature, score in top_factors:
                factors.append({
                    'feature': feature,
                    'importance': feature_importance.get(feature, 0),
                    'abnormality_score': float(normalized_values.get(feature, 0)),
                    'contribution_score': float(score),
                    'actual_value': float(feature_values.get(feature, 0))
                })
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error identifying contributing factors: {str(e)}")
            return []

    def _generate_anomaly_explanation(self,
                                    account_info: pd.Series,
                                    anomaly_types: List[str],
                                    contributing_factors: List[Dict],
                                    severity: str) -> str:
        """Generate detailed explanation of the anomaly"""
        try:
            explanation_parts = []
            
            # Account context
            explanation_parts.append(
                f"Account {account_info['account_number']} ({account_info['account_name']}) "
                f"has shown {severity.lower()} severity anomalous activity."
            )
            
            # Anomaly types
            type_descriptions = {
                'HIGH_VOLUME': 'unusually high transaction volume',
                'LOW_VOLUME': 'unusually low transaction volume',
                'UNUSUAL_BALANCE_CHANGE': 'significant balance changes',
                'IRREGULAR_TIMING': 'irregular transaction timing',
                'IRREGULAR_AMOUNTS': 'irregular transaction amounts',
                'UNUSUAL_POSITIVE_TREND': 'unusual positive trend',
                'UNUSUAL_NEGATIVE_TREND': 'unusual negative trend',
                'GENERAL_ANOMALY': 'generally anomalous behavior'
            }
            
            type_explanations = [
                type_descriptions.get(anomaly_type, anomaly_type.lower().replace('_', ' '))
                for anomaly_type in anomaly_types
            ]
            
            explanation_parts.append(
                f"The account has exhibited {', '.join(type_explanations)}."
            )
            
            # Contributing factors
            if contributing_factors:
                factor_explanations = []
                for factor in contributing_factors[:3]:  # Top 3 factors
                    factor_explanations.append(
                        f"{factor['feature'].replace('_', ' ').title()} "
                        f"(contribution score: {factor['contribution_score']:.2f})"
                    )
                
                explanation_parts.append(
                    f"Key contributing factors include: {', '.join(factor_explanations)}."
                )
            
            return ' '.join(explanation_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly explanation: {str(e)}")
            return "Error generating explanation"

    def _get_recommended_actions(self,
                               anomaly_types: List[str],
                               severity: str,
                               account_info: pd.Series) -> List[str]:
        """Generate recommended actions based on anomaly characteristics"""
        try:
            recommendations = []
            
            # Severity-based recommendations
            if severity == 'CRITICAL':
                recommendations.extend([
                    "Immediate review of account activity required",
                    "Consider temporary hold on automated transactions",
                    "Schedule urgent account audit"
                ])
            elif severity == 'HIGH':
                recommendations.extend([
                    "Review recent account activity within 24 hours",
                    "Increase monitoring frequency",
                    "Prepare detailed activity report"
                ])
            
            # Type-specific recommendations
            type_recommendations = {
                'HIGH_VOLUME': [
                    "Review transaction authorization limits",
                    "Verify transaction approvals"
                ],
                'LOW_VOLUME': [
                    "Check for missing transactions",
                    "Verify account status and access"
                ],
                'UNUSUAL_BALANCE_CHANGE': [
                    "Reconcile recent balance changes",
                    "Review large transactions"
                ],
                'IRREGULAR_TIMING': [
                    "Review transaction scheduling patterns",
                    "Check for automated transaction issues"
                ],
                'IRREGULAR_AMOUNTS': [
                    "Analyze transaction amount patterns",
                    "Review transaction categorization"
                ]
            }
            
            for anomaly_type in anomaly_types:
                if anomaly_type in type_recommendations:
                    recommendations.extend(type_recommendations[anomaly_type])
            
            # Account type specific recommendations
            account_type = account_info.get('account_type', '')
            if account_type == 'ASSET':
                recommendations.append("Review asset valuation and depreciation")
            elif account_type == 'LIABILITY':
                recommendations.append("Verify liability terms and payment schedules")
            elif account_type == 'REVENUE':
                recommendations.append("Review revenue recognition patterns")
            elif account_type == 'EXPENSE':
                recommendations.append("Analyze expense authorization process")
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review account activity and investigate anomalies"]

    def save_monitor_state(self, filepath: str) -> None:
        """Save trained monitor state to file"""
        try:
            monitor_state = {
                'isolation_forest': self.isolation_forest,
                'envelope_model': self.envelope_model,
                'lof_model': self.lof_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'training_period': self.training_period,
                'model_metrics': self.model_metrics,
                'monitor_params': self.monitor_params
            }
            
            joblib.dump(monitor_state, filepath)
            self.logger.info(f"Monitor state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving monitor state: {str(e)}")
            raise

    def load_monitor_state(self, filepath: str) -> None:
        """Load trained monitor state from file"""
        try:
            monitor_state = joblib.load(filepath)
            
            self.isolation_forest = monitor_state['isolation_forest']
            self.envelope_model = monitor_state['envelope_model']
            self.lof_model = monitor_state['lof_model']
            self.scaler = monitor_state['scaler']
            self.feature_columns = monitor_state['feature_columns']
            self.training_period = monitor_state['training_period']
            self.model_metrics = monitor_state['model_metrics']
            self.monitor_params = monitor_state['monitor_params']
            
            self.logger.info(f"Monitor state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading monitor state: {str(e)}")
            raise