# src/analytics/anomaly_detection/payment_pattern_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import joblib
import json
import os

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class PaymentPatternAnalyzer:
    """
    Analyzes and detects anomalies in payment patterns.
    Integrates with existing AP/AR data structure.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize analyzer with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.model_utils = ModelUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.dbscan_model = None
        self.isolation_forest = None
        self.scaler = None
        self.feature_columns = None
        self.training_period = None
        self.model_metrics = {}
        
        # Analysis parameters
        self.analysis_params = {
            'min_transactions': 5,  # Minimum transactions for pattern analysis
            'eps_quantile': 0.95,   # DBSCAN eps parameter quantile
            'min_samples': 3,       # DBSCAN minimum samples parameter
            'outlier_threshold': 3.0, # Standard deviations for outlier detection
            'time_window': 90       # Days for rolling pattern analysis
        }
        
    def prepare_pattern_data(self,
                           start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """
        Prepare data for payment pattern analysis
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            DataFrame with features for pattern analysis
        """
        try:
            # Get required data
            ap_data = self._get_ap_data(start_date, end_date)
            supplier_data = self._get_supplier_data()
            payment_history = self._get_payment_history(
                start_date - timedelta(days=self.analysis_params['time_window']),
                end_date
            )
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. Payment timing features
            timing_features = self._create_timing_features(ap_data)
            features_df = pd.concat([features_df, timing_features], axis=1)
            
            # 2. Payment amount features
            amount_features = self._create_amount_features(ap_data)
            features_df = pd.concat([features_df, amount_features], axis=1)
            
            # 3. Supplier features
            supplier_features = self._create_supplier_features(
                supplier_data, payment_history
            )
            features_df = pd.concat([features_df, supplier_features], axis=1)
            
            # 4. Historical pattern features
            pattern_features = self._create_pattern_features(payment_history)
            features_df = pd.concat([features_df, pattern_features], axis=1)
            
            # Store feature columns for later use
            self.feature_columns = features_df.columns.tolist()
            self.training_period = (start_date, end_date)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing pattern data: {str(e)}")
            raise

    def _get_ap_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get AP transaction data"""
        try:
            ap_invoices = self.data_loader.data['ap_invoices']
            ap_payments = self.data_loader.data['ap_payments']
            
            # Filter invoices for date range
            invoices = ap_invoices[
                ap_invoices['invoice_date'].between(start_date, end_date)
            ]
            
            # Filter payments for date range
            payments = ap_payments[
                ap_payments['payment_date'].between(start_date, end_date)
            ]
            
            # Join payments with invoices
            ap_data = pd.merge(
                invoices,
                payments[['invoice_id', 'payment_date', 'payment_method', 'amount']],
                on='invoice_id',
                how='left'
            )
            
            # Calculate payment delays
            ap_data['payment_delay'] = (
                ap_data['payment_date'] - ap_data['due_date']
            ).dt.days
            
            # Calculate payment ratios
            ap_data['payment_ratio'] = ap_data['amount_y'] / ap_data['amount_x']
            
            return ap_data
            
        except Exception as e:
            self.logger.error(f"Error getting AP data: {str(e)}")
            raise

    def _get_supplier_data(self) -> pd.DataFrame:
        """Get supplier master data"""
        try:
            suppliers = self.data_loader.data['ap_suppliers']
            
            # Get active suppliers
            supplier_data = suppliers[
                suppliers['status'] == 'ACTIVE'
            ].copy()
            
            return supplier_data
            
        except Exception as e:
            self.logger.error(f"Error getting supplier data: {str(e)}")
            raise

    def _get_payment_history(self,
                           start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """Get historical payment data"""
        try:
            ap_payments = self.data_loader.data['ap_payments']
            ap_invoices = self.data_loader.data['ap_invoices']
            
            # Get payment history
            payment_history = pd.merge(
                ap_payments[
                    ap_payments['payment_date'].between(start_date, end_date)
                ],
                ap_invoices[['invoice_id', 'invoice_date', 'due_date', 'amount']],
                on='invoice_id'
            )
            
            # Calculate additional fields
            payment_history['payment_delay'] = (
                payment_history['payment_date'] - payment_history['due_date']
            ).dt.days
            
            payment_history['days_to_pay'] = (
                payment_history['payment_date'] - payment_history['invoice_date']
            ).dt.days
            
            payment_history['payment_ratio'] = (
                payment_history['amount_y'] / payment_history['amount_x']
            )
            
            return payment_history
            
        except Exception as e:
            self.logger.error(f"Error getting payment history: {str(e)}")
            raise


    

    def _create_timing_features(self, ap_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on payment timing"""
        try:
            # Group by supplier and payment date
            timing_features = pd.DataFrame()
            
            # Payment delay metrics
            timing_features['avg_payment_delay'] = ap_data.groupby('supplier_id')['payment_delay'].mean()
            timing_features['std_payment_delay'] = ap_data.groupby('supplier_id')['payment_delay'].std()
            timing_features['max_payment_delay'] = ap_data.groupby('supplier_id')['payment_delay'].max()
            
            # Payment consistency metrics
            timing_features['delay_variability'] = (
                timing_features['std_payment_delay'] /
                timing_features['avg_payment_delay']
            ).fillna(0)
            
            # Late payment metrics
            timing_features['late_payment_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: (x['payment_delay'] > 0).mean()
            )
            
            # Early payment metrics
            timing_features['early_payment_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: (x['payment_delay'] < 0).mean()
            )
            
            # Payment timing patterns
            payment_dow = ap_data['payment_date'].dt.dayofweek
            payment_hour = ap_data['payment_date'].dt.hour
            
            timing_features['weekend_payment_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: (x['payment_date'].dt.dayofweek.isin([5, 6])).mean()
            )
            
            timing_features['nonbusiness_hour_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: ((x['payment_date'].dt.hour < 9) | 
                         (x['payment_date'].dt.hour > 17)).mean()
            )
            
            # Payment interval metrics
            timing_features['avg_payment_interval'] = ap_data.groupby('supplier_id').apply(
                lambda x: x['payment_date'].diff().dt.days.mean()
            )
            
            timing_features['std_payment_interval'] = ap_data.groupby('supplier_id').apply(
                lambda x: x['payment_date'].diff().dt.days.std()
            )
            
            return timing_features
            
        except Exception as e:
            self.logger.error(f"Error creating timing features: {str(e)}")
            raise

    def _create_amount_features(self, ap_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on payment amounts"""
        try:
            # Group by supplier
            amount_features = pd.DataFrame()
            
            # Basic amount statistics
            amount_features['avg_payment_amount'] = ap_data.groupby('supplier_id')['amount_y'].mean()
            amount_features['std_payment_amount'] = ap_data.groupby('supplier_id')['amount_y'].std()
            amount_features['max_payment_amount'] = ap_data.groupby('supplier_id')['amount_y'].max()
            
            # Payment amount variability
            amount_features['amount_variability'] = (
                amount_features['std_payment_amount'] /
                amount_features['avg_payment_amount']
            ).fillna(0)
            
            # Payment ratio metrics
            amount_features['avg_payment_ratio'] = ap_data.groupby('supplier_id')['payment_ratio'].mean()
            amount_features['std_payment_ratio'] = ap_data.groupby('supplier_id')['payment_ratio'].std()
            
            # Overpayment/underpayment metrics
            amount_features['overpayment_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: (x['payment_ratio'] > 1.0).mean()
            )
            
            amount_features['underpayment_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: (x['payment_ratio'] < 1.0).mean()
            )
            
            # Round amounts analysis
            amount_features['round_amount_rate'] = ap_data.groupby('supplier_id').apply(
                lambda x: (x['amount_y'] % 100 == 0).mean()
            )
            
            # Payment clustering
            amount_features['amount_cluster_ratio'] = ap_data.groupby('supplier_id').apply(
                lambda x: len(pd.qcut(x['amount_y'], q=4, duplicates='drop'))
            ) / ap_data.groupby('supplier_id').size()
            
            return amount_features
            
        except Exception as e:
            self.logger.error(f"Error creating amount features: {str(e)}")
            raise

    def _create_supplier_features(self,
                                supplier_data: pd.DataFrame,
                                payment_history: pd.DataFrame) -> pd.DataFrame:
        """Create features based on supplier characteristics"""
        try:
            # Initialize supplier features
            supplier_features = pd.DataFrame()
            
            # Payment terms analysis
            supplier_features['payment_terms'] = pd.get_dummies(
                supplier_data['payment_terms'],
                prefix='terms'
            )
            
            # Calculate supplier age
            supplier_features['supplier_age_days'] = (
                datetime.now() - pd.to_datetime(supplier_data['created_date'])
            ).dt.days
            
            # Payment method diversity
            supplier_features['payment_method_count'] = payment_history.groupby('supplier_id')[
                'payment_method'
            ].nunique()
            
            # Transaction frequency
            supplier_features['transaction_frequency'] = (
                payment_history.groupby('supplier_id').size() /
                (payment_history['payment_date'].max() -
                 payment_history['payment_date'].min()).days * 30  # Monthly frequency
            )
            
            # Payment regularity score
            supplier_features['payment_regularity'] = payment_history.groupby('supplier_id').apply(
                self._calculate_regularity_score
            )
            
            # Recent activity metrics
            recent_cutoff = datetime.now() - timedelta(days=30)
            supplier_features['recent_activity_ratio'] = payment_history.groupby('supplier_id').apply(
                lambda x: (x['payment_date'] > recent_cutoff).mean()
            )
            
            return supplier_features
            
        except Exception as e:
            self.logger.error(f"Error creating supplier features: {str(e)}")
            raise

    def _calculate_regularity_score(self, group: pd.DataFrame) -> float:
        """Calculate payment regularity score for a supplier"""
        try:
            if len(group) < 2:
                return 0.0
                
            # Calculate payment intervals
            payment_intervals = group['payment_date'].diff().dt.days
            
            # Calculate coefficient of variation
            cv = payment_intervals.std() / payment_intervals.mean() if payment_intervals.mean() != 0 else 0
            
            # Convert to regularity score (0 to 1, higher is more regular)
            regularity_score = 1 / (1 + cv)
            
            return regularity_score
            
        except Exception as e:
            self.logger.error(f"Error calculating regularity score: {str(e)}")
            return 0.0

    def _create_pattern_features(self, payment_history: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical payment patterns"""
        try:
            # Initialize pattern features
            pattern_features = pd.DataFrame()
            
            # Temporal pattern metrics
            pattern_features['day_of_week_entropy'] = payment_history.groupby('supplier_id').apply(
                lambda x: self._calculate_entropy(x['payment_date'].dt.dayofweek)
            )
            
            pattern_features['day_of_month_entropy'] = payment_history.groupby('supplier_id').apply(
                lambda x: self._calculate_entropy(x['payment_date'].dt.day)
            )
            
            # Payment behavior consistency
            pattern_features['behavior_consistency'] = payment_history.groupby('supplier_id').apply(
                self._calculate_behavior_consistency
            )
            
            # Pattern complexity metrics
            pattern_features['pattern_complexity'] = payment_history.groupby('supplier_id').apply(
                self._calculate_pattern_complexity
            )
            
            # Seasonal pattern strength
            pattern_features['seasonal_strength'] = payment_history.groupby('supplier_id').apply(
                self._calculate_seasonal_strength
            )
            
            # Trend metrics
            pattern_features['trend_strength'] = payment_history.groupby('supplier_id').apply(
                self._calculate_trend_strength
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

    def _calculate_behavior_consistency(self, group: pd.DataFrame) -> float:
        """Calculate payment behavior consistency score"""
        try:
            if len(group) < self.analysis_params['min_transactions']:
                return 0.0
                
            # Calculate consistency metrics
            delay_consistency = 1 / (1 + group['payment_delay'].std())
            amount_consistency = 1 / (1 + group['payment_ratio'].std())
            interval_consistency = 1 / (1 + group['payment_date'].diff().dt.days.std())
            
            # Combine into overall score
            consistency_score = np.mean([
                delay_consistency,
                amount_consistency,
                interval_consistency
            ])
            
            return float(consistency_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating behavior consistency: {str(e)}")
            return 0.0

    def _calculate_pattern_complexity(self, group: pd.DataFrame) -> float:
        """Calculate complexity of payment patterns"""
        try:
            if len(group) < self.analysis_params['min_transactions']:
                return 0.0
                
            # Calculate various complexity metrics
            timing_complexity = self._calculate_entropy(group['payment_date'].dt.dayofweek)
            amount_complexity = len(pd.qcut(group['amount_y'], q=4, duplicates='drop'))
            method_complexity = group['payment_method'].nunique()
            
            # Combine into complexity score
            complexity_score = np.mean([
                timing_complexity / 2.8,  # Normalize by max entropy for 7 days
                amount_complexity / 4.0,  # Normalize by number of quartiles
                method_complexity / 3.0   # Normalize by typical number of payment methods
            ])
            
            return float(complexity_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern complexity: {str(e)}")
            return 0.0

    def _calculate_seasonal_strength(self, group: pd.DataFrame) -> float:
        """Calculate strength of seasonal patterns"""
        try:
            if len(group) < self.analysis_params['min_transactions']:
                return 0.0
                
            # Calculate monthly payment frequencies
            monthly_freq = group.groupby(group['payment_date'].dt.month).size()
            
            # Calculate coefficient of variation of monthly frequencies
            cv = monthly_freq.std() / monthly_freq.mean() if monthly_freq.mean() != 0 else 0
            
            # Convert to seasonal strength score
            seasonal_strength = 1 / (1 + cv)
            
            return float(seasonal_strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal strength: {str(e)}")
            return 0.0

    def _calculate_trend_strength(self, group: pd.DataFrame) -> float:
        """Calculate strength of payment trends"""
        try:
            if len(group) < self.analysis_params['min_transactions']:
                return 0.0
                
            # Calculate trend in payment amounts
            amounts = group['amount_y'].values
            time_index = np.arange(len(amounts))
            
            # Fit linear trend
            slope, _ = np.polyfit(time_index, amounts, 1)
            
            # Calculate R-squared
            trend_line = slope * time_index
            ss_tot = np.sum((amounts - amounts.mean()) ** 2)
            ss_res = np.sum((amounts - trend_line) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return float(abs(r_squared))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
        
    
    

    def analyze_payment_patterns(self,
                               pattern_data: pd.DataFrame,
                               contamination: float = 0.1) -> Dict:
        """
        Analyze payment patterns using multiple detection methods
        
        Args:
            pattern_data: DataFrame with pattern features
            contamination: Expected proportion of anomalies
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(pattern_data)
            
            # Detect clusters using DBSCAN
            self.dbscan_model = DBSCAN(
                eps=self._estimate_eps(X_scaled),
                min_samples=self.analysis_params['min_samples']
            )
            cluster_labels = self.dbscan_model.fit_predict(X_scaled)
            
            # Detect anomalies using Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
            
            # Calculate pattern metrics
            pattern_metrics = self._calculate_pattern_metrics(
                pattern_data,
                cluster_labels,
                anomaly_labels
            )
            
            # Store results
            self.model_metrics = pattern_metrics
            return pattern_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing payment patterns: {str(e)}")
            raise

    def _estimate_eps(self, X_scaled: np.ndarray) -> float:
        """Estimate DBSCAN eps parameter using nearest neighbors"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Calculate distances to k-nearest neighbors
            k = self.analysis_params['min_samples']
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            
            # Sort distances to kth neighbor
            k_distances = np.sort(distances[:, -1])
            
            # Estimate eps using specified quantile
            eps = np.quantile(k_distances, self.analysis_params['eps_quantile'])
            
            return float(eps)
            
        except Exception as e:
            self.logger.error(f"Error estimating eps parameter: {str(e)}")
            raise

    def _calculate_pattern_metrics(self,
                                 pattern_data: pd.DataFrame,
                                 cluster_labels: np.ndarray,
                                 anomaly_labels: np.ndarray) -> Dict:
        """Calculate comprehensive pattern analysis metrics"""
        try:
            # Cluster analysis metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_points = np.sum(cluster_labels == -1)
            
            cluster_metrics = {
                'n_clusters': n_clusters,
                'noise_ratio': noise_points / len(cluster_labels),
                'cluster_sizes': {
                    f'cluster_{i}': int(np.sum(cluster_labels == i))
                    for i in range(n_clusters)
                },
                'cluster_stats': self._calculate_cluster_statistics(
                    pattern_data, cluster_labels
                )
            }
            
            # Anomaly detection metrics
            anomaly_metrics = {
                'anomaly_count': int(np.sum(anomaly_labels == -1)),
                'anomaly_rate': float(np.mean(anomaly_labels == -1) * 100),
                'anomaly_scores': self._calculate_anomaly_scores(
                    pattern_data, anomaly_labels
                )
            }
            
            # Pattern profile metrics
            pattern_profiles = self._create_pattern_profiles(
                pattern_data, cluster_labels, anomaly_labels
            )
            
            return {
                'cluster_metrics': cluster_metrics,
                'anomaly_metrics': anomaly_metrics,
                'pattern_profiles': pattern_profiles,
                'analysis_params': {
                    'eps': self.dbscan_model.eps,
                    'min_samples': self.dbscan_model.min_samples,
                    'contamination': self.isolation_forest.contamination
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern metrics: {str(e)}")
            raise

    def _calculate_cluster_statistics(self,
                                   pattern_data: pd.DataFrame,
                                   cluster_labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster"""
        try:
            cluster_stats = {}
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = pattern_data[cluster_mask]
                
                stats = {
                    'size': int(np.sum(cluster_mask)),
                    'centroid': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict(),
                    'feature_importance': self._calculate_cluster_feature_importance(
                        pattern_data, cluster_mask
                    )
                }
                
                cluster_stats[f'cluster_{i}'] = stats
                
            return cluster_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster statistics: {str(e)}")
            raise

    def _calculate_cluster_feature_importance(self,
                                          data: pd.DataFrame,
                                          cluster_mask: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for cluster characterization"""
        try:
            feature_importance = {}
            
            for feature in data.columns:
                # Calculate separation metric
                in_cluster_mean = data[feature][cluster_mask].mean()
                out_cluster_mean = data[feature][~cluster_mask].mean()
                in_cluster_std = data[feature][cluster_mask].std()
                out_cluster_std = data[feature][~cluster_mask].std()
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((in_cluster_std**2 + out_cluster_std**2) / 2)
                effect_size = abs(in_cluster_mean - out_cluster_mean) / pooled_std if pooled_std != 0 else 0
                
                feature_importance[feature] = float(effect_size)
                
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster feature importance: {str(e)}")
            raise

    def _calculate_anomaly_scores(self,
                                pattern_data: pd.DataFrame,
                                anomaly_labels: np.ndarray) -> Dict:
        """Calculate detailed anomaly scores"""
        try:
            # Get anomaly scores from Isolation Forest
            anomaly_scores = -self.isolation_forest.score_samples(
                self.scaler.transform(pattern_data)
            )
            
            # Calculate score statistics
            score_stats = {
                'min_score': float(np.min(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores)),
                'mean_score': float(np.mean(anomaly_scores)),
                'std_score': float(np.std(anomaly_scores)),
                'score_percentiles': {
                    str(p): float(np.percentile(anomaly_scores, p))
                    for p in [25, 50, 75, 90, 95, 99]
                }
            }
            
            # Calculate feature contributions
            feature_contributions = self._calculate_feature_contributions(
                pattern_data,
                anomaly_scores
            )
            
            return {
                'score_statistics': score_stats,
                'feature_contributions': feature_contributions
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly scores: {str(e)}")
            raise

    def _calculate_feature_contributions(self,
                                      pattern_data: pd.DataFrame,
                                      anomaly_scores: np.ndarray) -> Dict[str, float]:
        """Calculate feature contributions to anomaly scores"""
        try:
            feature_contributions = {}
            
            for feature in pattern_data.columns:
                # Calculate correlation with anomaly scores
                correlation = np.corrcoef(pattern_data[feature], anomaly_scores)[0, 1]
                
                # Calculate feature importance using random forest
                feature_importance = np.abs(
                    self.isolation_forest.feature_importances_[
                        list(pattern_data.columns).index(feature)
                    ]
                )
                
                # Combine metrics
                contribution_score = np.mean([abs(correlation), feature_importance])
                feature_contributions[feature] = float(contribution_score)
                
            return feature_contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating feature contributions: {str(e)}")
            raise

    def _create_pattern_profiles(self,
                               pattern_data: pd.DataFrame,
                               cluster_labels: np.ndarray,
                               anomaly_labels: np.ndarray) -> Dict:
        """Create profiles of identified payment patterns"""
        try:
            profiles = {}
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Create cluster profiles
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = pattern_data[cluster_mask]
                
                profiles[f'cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'characteristics': self._extract_pattern_characteristics(cluster_data),
                    'anomaly_rate': float(
                        np.mean(anomaly_labels[cluster_mask] == -1) * 100
                    )
                }
                
            # Create anomaly profile
            anomaly_mask = anomaly_labels == -1
            anomaly_data = pattern_data[anomaly_mask]
            
            profiles['anomalies'] = {
                'count': int(np.sum(anomaly_mask)),
                'characteristics': self._extract_pattern_characteristics(anomaly_data),
                'cluster_distribution': {
                    f'cluster_{i}': int(np.sum(cluster_labels[anomaly_mask] == i))
                    for i in range(n_clusters)
                }
            }
            
            return profiles
            
        except Exception as e:
            self.logger.error(f"Error creating pattern profiles: {str(e)}")
            raise


    

    def _extract_pattern_characteristics(self, pattern_data: pd.DataFrame) -> Dict:
        """Extract key characteristics of payment patterns"""
        try:
            characteristics = {
                'timing_patterns': self._extract_timing_characteristics(pattern_data),
                'amount_patterns': self._extract_amount_characteristics(pattern_data),
                'behavior_patterns': self._extract_behavior_characteristics(pattern_data)
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern characteristics: {str(e)}")
            raise

    def _extract_timing_characteristics(self, pattern_data: pd.DataFrame) -> Dict:
        """Extract timing-related pattern characteristics"""
        try:
            timing_chars = {
                'avg_payment_delay': float(pattern_data['avg_payment_delay'].mean()),
                'payment_delay_variability': float(pattern_data['delay_variability'].mean()),
                'late_payment_tendency': float(pattern_data['late_payment_rate'].mean() * 100),
                'payment_regularity': float(pattern_data['payment_regularity'].mean()),
                'common_patterns': {
                    'weekend_payments': float(pattern_data['weekend_payment_rate'].mean() * 100),
                    'nonbusiness_hours': float(pattern_data['nonbusiness_hour_rate'].mean() * 100),
                    'early_payments': float(pattern_data['early_payment_rate'].mean() * 100)
                }
            }
            
            return timing_chars
            
        except Exception as e:
            self.logger.error(f"Error extracting timing characteristics: {str(e)}")
            raise

    def _extract_amount_characteristics(self, pattern_data: pd.DataFrame) -> Dict:
        """Extract amount-related pattern characteristics"""
        try:
            amount_chars = {
                'amount_variability': float(pattern_data['amount_variability'].mean()),
                'payment_ratio_consistency': float(1 - pattern_data['std_payment_ratio'].mean()),
                'payment_tendencies': {
                    'overpayment_frequency': float(pattern_data['overpayment_rate'].mean() * 100),
                    'underpayment_frequency': float(pattern_data['underpayment_rate'].mean() * 100),
                    'round_amount_frequency': float(pattern_data['round_amount_rate'].mean() * 100)
                },
                'amount_clustering': float(pattern_data['amount_cluster_ratio'].mean())
            }
            
            return amount_chars
            
        except Exception as e:
            self.logger.error(f"Error extracting amount characteristics: {str(e)}")
            raise

    def _extract_behavior_characteristics(self, pattern_data: pd.DataFrame) -> Dict:
        """Extract behavioral pattern characteristics"""
        try:
            behavior_chars = {
                'consistency': {
                    'timing_consistency': float(pattern_data['payment_regularity'].mean()),
                    'amount_consistency': float(1 - pattern_data['amount_variability'].mean()),
                    'behavior_consistency': float(pattern_data['behavior_consistency'].mean())
                },
                'complexity': {
                    'pattern_complexity': float(pattern_data['pattern_complexity'].mean()),
                    'payment_method_diversity': float(pattern_data['payment_method_count'].mean())
                },
                'trends': {
                    'seasonal_strength': float(pattern_data['seasonal_strength'].mean()),
                    'trend_strength': float(pattern_data['trend_strength'].mean())
                }
            }
            
            return behavior_chars
            
        except Exception as e:
            self.logger.error(f"Error extracting behavior characteristics: {str(e)}")
            raise

    def analyze_supplier_patterns(self, supplier_id: int) -> Dict:
        """
        Analyze payment patterns for a specific supplier
        
        Args:
            supplier_id: Supplier ID to analyze
            
        Returns:
            Dict containing supplier pattern analysis
        """
        try:
            # Get supplier data
            supplier_data = self._get_supplier_data()
            supplier_info = supplier_data[
                supplier_data['supplier_id'] == supplier_id
            ]
            
            if supplier_info.empty:
                raise ValueError(f"Supplier {supplier_id} not found")
                
            # Get payment history
            payment_history = self._get_payment_history(
                datetime.now() - timedelta(days=365),
                datetime.now()
            )
            supplier_history = payment_history[
                payment_history['supplier_id'] == supplier_id
            ]
            
            if len(supplier_history) < self.analysis_params['min_transactions']:
                return {
                    'supplier_id': supplier_id,
                    'status': 'INSUFFICIENT_DATA',
                    'min_transactions': self.analysis_params['min_transactions']
                }
            
            # Calculate pattern metrics
            pattern_metrics = {
                'timing_metrics': self._analyze_supplier_timing(supplier_history),
                'amount_metrics': self._analyze_supplier_amounts(supplier_history),
                'behavior_metrics': self._analyze_supplier_behavior(supplier_history)
            }
            
            # Detect pattern changes
            pattern_changes = self._detect_pattern_changes(supplier_history)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_supplier_risk(
                pattern_metrics,
                pattern_changes
            )
            
            return {
                'supplier_id': supplier_id,
                'status': 'ANALYZED',
                'pattern_metrics': pattern_metrics,
                'pattern_changes': pattern_changes,
                'risk_metrics': risk_metrics,
                'recommendations': self._generate_recommendations(risk_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing supplier patterns: {str(e)}")
            raise

    def _detect_pattern_changes(self, payment_history: pd.DataFrame) -> Dict:
        """Detect changes in payment patterns"""
        try:
            # Split history into recent and historical periods
            midpoint = len(payment_history) // 2
            historical = payment_history.iloc[:midpoint]
            recent = payment_history.iloc[midpoint:]
            
            changes = {
                'timing_changes': self._compare_periods(
                    historical, recent, ['payment_delay', 'days_to_pay']
                ),
                'amount_changes': self._compare_periods(
                    historical, recent, ['amount_y', 'payment_ratio']
                ),
                'pattern_changes': {
                    'regularity_change': self._calculate_regularity_score(recent) -
                                      self._calculate_regularity_score(historical),
                    'complexity_change': self._calculate_pattern_complexity(recent) -
                                      self._calculate_pattern_complexity(historical)
                }
            }
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern changes: {str(e)}")
            raise

    def _compare_periods(self,
                        historical: pd.DataFrame,
                        recent: pd.DataFrame,
                        metrics: List[str]) -> Dict:
        """Compare metrics between historical and recent periods"""
        try:
            changes = {}
            
            for metric in metrics:
                historical_stats = historical[metric].describe()
                recent_stats = recent[metric].describe()
                
                changes[f'{metric}_change'] = {
                    'mean_change': float(
                        (recent_stats['mean'] - historical_stats['mean']) /
                        historical_stats['mean'] * 100
                    ),
                    'variability_change': float(
                        (recent_stats['std'] - historical_stats['std']) /
                        historical_stats['std'] * 100
                    )
                }
                
            return changes
            
        except Exception as e:
            self.logger.error(f"Error comparing periods: {str(e)}")
            raise

    def _calculate_supplier_risk(self,
                               pattern_metrics: Dict,
                               pattern_changes: Dict) -> Dict:
        """Calculate supplier risk metrics"""
        try:
            risk_metrics = {
                'timing_risk': self._calculate_timing_risk(
                    pattern_metrics['timing_metrics'],
                    pattern_changes['timing_changes']
                ),
                'amount_risk': self._calculate_amount_risk(
                    pattern_metrics['amount_metrics'],
                    pattern_changes['amount_changes']
                ),
                'behavior_risk': self._calculate_behavior_risk(
                    pattern_metrics['behavior_metrics'],
                    pattern_changes['pattern_changes']
                )
            }
            
            # Calculate composite risk score
            risk_scores = [
                risk_metrics['timing_risk']['risk_score'],
                risk_metrics['amount_risk']['risk_score'],
                risk_metrics['behavior_risk']['risk_score']
            ]
            
            risk_metrics['composite_risk'] = {
                'risk_score': float(np.mean(risk_scores)),
                'risk_level': self._determine_risk_level(np.mean(risk_scores))
            }
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating supplier risk: {str(e)}")
            raise

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_recommendations(self, risk_metrics: Dict) -> List[str]:
        """Generate recommendations based on risk analysis"""
        recommendations = []
        
        if risk_metrics['composite_risk']['risk_level'] == 'CRITICAL':
            recommendations.extend([
                "Immediate review of payment patterns required",
                "Consider payment term adjustments",
                "Implement strict payment monitoring"
            ])
        elif risk_metrics['composite_risk']['risk_level'] == 'HIGH':
            recommendations.extend([
                "Review payment history and patterns",
                "Monitor for further pattern changes",
                "Consider payment behavior analysis"
            ])
        elif risk_metrics['composite_risk']['risk_level'] == 'MEDIUM':
            recommendations.extend([
                "Regular monitoring of payment patterns",
                "Document pattern changes",
                "Review if patterns deteriorate"
            ])
        else:
            recommendations.extend([
                "Maintain current monitoring practices",
                "Document good payment patterns"
            ])
            
        return recommendations

    def save_analyzer(self, model_dir: str) -> None:
        """Save trained analyzer models"""
        try:
            if self.dbscan_model is None or self.isolation_forest is None:
                raise ValueError("Models have not been trained yet")
                
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            joblib.dump(self.dbscan_model,
                       os.path.join(model_dir, 'dbscan_model.joblib'))
            joblib.dump(self.isolation_forest,
                       os.path.join(model_dir, 'isolation_forest.joblib'))
            joblib.dump(self.scaler,
                       os.path.join(model_dir, 'scaler.joblib'))
            
            # Save model info
            model_info = {
                'feature_columns': self.feature_columns,
                'training_period': self.training_period,
                'model_metrics': self.model_metrics,
                'analysis_params': self.analysis_params,
                'save_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, default=str)
                
            self.logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def load_analyzer(self, model_dir: str) -> None:
        """Load saved analyzer models"""
        try:
            # Load models
            self.dbscan_model = joblib.load(
                os.path.join(model_dir, 'dbscan_model.joblib')
            )
            self.isolation_forest = joblib.load(
                os.path.join(model_dir, 'isolation_forest.joblib')
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
            self.analysis_params = model_info['analysis_params']
            
            self.logger.info(f"Models loaded from {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise