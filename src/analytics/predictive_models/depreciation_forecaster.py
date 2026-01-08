# src/analytics/predictive_models/depreciation_forecaster.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import json
import os

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class DepreciationForecaster:
    """
    Predicts asset depreciation using historical FA data.
    Integrates with existing fixed assets and GL data structure.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize forecaster with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.model_utils = ModelUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_period = None
        self.model_metrics = {}
        
        # Define depreciation methods supported by the system
        self.depreciation_methods = [
            'STRAIGHT_LINE',
            'DECLINING_BALANCE',
            'DOUBLE_DECLINING',
            'SUM_OF_YEARS_DIGITS',
            'UNITS_OF_PRODUCTION'
        ]
        
    def prepare_training_data(self,
                            start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
        """
        Prepare training data for depreciation forecasting
        
        Args:
            start_date: Start of training period
            end_date: End of training period
            
        Returns:
            DataFrame with features and target
        """
        try:
            # Get required data
            fa_data = self._get_fa_data(start_date, end_date)
            depreciation_history = self._get_depreciation_history(start_date, end_date)
            gl_data = self._get_gl_data(start_date, end_date)
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. Asset features
            asset_features = self._create_asset_features(fa_data)
            features_df = pd.concat([features_df, asset_features], axis=1)
            
            # 2. Depreciation history features
            depreciation_features = self._create_depreciation_features(depreciation_history)
            features_df = pd.concat([features_df, depreciation_features], axis=1)
            
            # 3. GL transaction features
            gl_features = self._create_gl_features(gl_data)
            features_df = pd.concat([features_df, gl_features], axis=1)
            
            # 4. Time-based features
            time_features = self._create_time_features(features_df.index)
            features_df = pd.concat([features_df, time_features], axis=1)
            
            # Create target variables (next period depreciation)
            target = self._create_depreciation_target(depreciation_history)
            
            # Combine features and target
            training_data = pd.concat([features_df, target.rename('target')], axis=1)
            training_data = training_data.dropna()  # Remove any rows with missing values
            
            # Store feature columns for later use
            self.feature_columns = features_df.columns.tolist()
            self.training_period = (start_date, end_date)
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def _get_fa_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get fixed asset data"""
        try:
            fa_assets = self.data_loader.data['fa_assets']
            fa_categories = self.data_loader.data['fa_categories']
            
            # Filter assets based on acquisition date
            fa_data = fa_assets[
                (fa_assets['acquisition_date'] <= end_date) &
                (fa_assets['status'] == 'IN_SERVICE')
            ]
            
            # Join with categories
            fa_data = pd.merge(
                fa_data,
                fa_categories[['category_id', 'category_name', 'depreciation_method']],
                on='category_id',
                how='left'
            )
            
            return fa_data
            
        except Exception as e:
            self.logger.error(f"Error getting FA data: {str(e)}")
            raise

    def _get_depreciation_history(self, 
                                start_date: datetime,
                                end_date: datetime) -> pd.DataFrame:
        """Get historical depreciation data"""
        try:
            fa_depreciation = self.data_loader.data['fa_depreciation']
            
            # Filter for date range
            depreciation_history = fa_depreciation[
                fa_depreciation['period_date'].between(start_date, end_date)
            ].copy()
            
            # Calculate additional fields
            depreciation_history['depreciation_rate'] = (
                depreciation_history['depreciation_amount'] /
                depreciation_history['book_value'].shift(1)
            )
            
            return depreciation_history
            
        except Exception as e:
            self.logger.error(f"Error getting depreciation history: {str(e)}")
            raise

    def _get_gl_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get relevant GL transactions for fixed assets"""
        try:
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_coa = self.data_loader.data['gl_coa']
            
            # Get fixed asset accounts
            fa_accounts = gl_coa[
                (gl_coa['account_type'] == 'ASSET') &
                (gl_coa['account_category'] == 'FIXED')
            ]['account_id'].tolist()
            
            # Filter relevant transactions
            gl_data = gl_lines[
                (gl_lines['account_id'].isin(fa_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(start_date, end_date)) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            return gl_data
            
        except Exception as e:
            self.logger.error(f"Error getting GL data: {str(e)}")
            raise

    

    def _create_asset_features(self, fa_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from fixed asset data"""
        try:
            # Initialize features DataFrame
            asset_features = pd.DataFrame()
            
            # Basic asset features
            asset_features['acquisition_cost'] = fa_data['acquisition_cost']
            asset_features['salvage_value'] = fa_data['salvage_value']
            asset_features['life_years'] = fa_data['life_years']
            
            # Calculate asset age
            asset_features['asset_age'] = (
                datetime.now() - fa_data['acquisition_date']
            ).dt.days / 365.25  # Convert to years
            
            # Calculate remaining life
            asset_features['remaining_life'] = fa_data['life_years'] - asset_features['asset_age']
            asset_features['life_used_percent'] = (
                asset_features['asset_age'] / fa_data['life_years'] * 100
            )
            
            # Depreciation method features (one-hot encoding)
            method_dummies = pd.get_dummies(
                fa_data['depreciation_method'],
                prefix='depreciation_method'
            )
            asset_features = pd.concat([asset_features, method_dummies], axis=1)
            
            # Category features
            category_dummies = pd.get_dummies(
                fa_data['category_name'],
                prefix='category'
            )
            asset_features = pd.concat([asset_features, category_dummies], axis=1)
            
            # Calculate value metrics
            asset_features['cost_to_salvage_ratio'] = (
                fa_data['acquisition_cost'] / fa_data['salvage_value']
            ).fillna(0)
            
            asset_features['annual_depreciation_rate'] = (
                (fa_data['acquisition_cost'] - fa_data['salvage_value']) /
                (fa_data['acquisition_cost'] * fa_data['life_years'])
            ).fillna(0)
            
            return asset_features
            
        except Exception as e:
            self.logger.error(f"Error creating asset features: {str(e)}")
            raise

    def _create_depreciation_features(self, depreciation_history: pd.DataFrame) -> pd.DataFrame:
        """Create features from depreciation history"""
        try:
            # Group by asset and period
            depreciation_features = depreciation_history.groupby(['asset_id', 'period_date']).agg({
                'depreciation_amount': ['mean', 'std', 'sum'],
                'accumulated_depreciation': 'last',
                'book_value': 'last',
                'depreciation_rate': 'mean'
            })
            
            # Flatten column names
            depreciation_features.columns = [
                'avg_depreciation',
                'std_depreciation',
                'total_depreciation',
                'accumulated_depreciation',
                'book_value',
                'avg_depreciation_rate'
            ]
            
            # Calculate additional metrics
            depreciation_features['depreciation_to_book_ratio'] = (
                depreciation_features['total_depreciation'] /
                depreciation_features['book_value']
            ).fillna(0)
            
            # Calculate trending metrics
            for window in [3, 6, 12]:
                # Rolling averages
                depreciation_features[f'rolling_{window}m_avg_depreciation'] = (
                    depreciation_features['avg_depreciation']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                # Rolling standard deviations
                depreciation_features[f'rolling_{window}m_std_depreciation'] = (
                    depreciation_features['avg_depreciation']
                    .rolling(window=window, min_periods=1)
                    .std()
                )
                
                # Rolling depreciation rates
                depreciation_features[f'rolling_{window}m_depreciation_rate'] = (
                    depreciation_features['avg_depreciation_rate']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            
            # Calculate momentum indicators
            depreciation_features['depreciation_momentum_3m'] = (
                depreciation_features['avg_depreciation'] -
                depreciation_features['rolling_3m_avg_depreciation']
            )
            
            depreciation_features['depreciation_momentum_6m'] = (
                depreciation_features['avg_depreciation'] -
                depreciation_features['rolling_6m_avg_depreciation']
            )
            
            return depreciation_features
            
        except Exception as e:
            self.logger.error(f"Error creating depreciation features: {str(e)}")
            raise

    def _create_gl_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from GL transactions"""
        try:
            # Group by account and period
            gl_features = gl_data.groupby(['account_id', 'journal_date']).agg({
                'debit_amount': ['sum', 'count', 'mean', 'std'],
                'credit_amount': ['sum', 'count', 'mean', 'std']
            })
            
            # Flatten column names
            gl_features.columns = [
                'total_debits',
                'debit_transactions',
                'avg_debit',
                'std_debit',
                'total_credits',
                'credit_transactions',
                'avg_credit',
                'std_credit'
            ]
            
            # Calculate additional metrics
            gl_features['net_movement'] = gl_features['total_debits'] - gl_features['total_credits']
            gl_features['transaction_volume'] = (
                gl_features['debit_transactions'] + gl_features['credit_transactions']
            )
            
            # Calculate transaction intensity
            gl_features['transaction_intensity'] = (
                gl_features['transaction_volume'] /
                gl_features['transaction_volume'].mean()
            )
            
            # Add rolling metrics
            for window in [30, 90]:
                gl_features[f'rolling_{window}d_net_movement'] = (
                    gl_features['net_movement']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                gl_features[f'rolling_{window}d_volume'] = (
                    gl_features['transaction_volume']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            
            return gl_features
            
        except Exception as e:
            self.logger.error(f"Error creating GL features: {str(e)}")
            raise

    def _create_time_features(self, dates_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-based features"""
        try:
            features = pd.DataFrame(index=dates_index)
            
            # Basic time features
            features['month'] = dates_index.month
            features['quarter'] = dates_index.quarter
            features['year'] = dates_index.year
            features['day_of_month'] = dates_index.day
            
            # Fiscal period indicators
            features['is_month_end'] = dates_index.is_month_end
            features['is_quarter_end'] = dates_index.is_quarter_end
            features['is_year_end'] = dates_index.is_year_end
            
            # Cyclic encoding of time features
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
            features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
            
            # Add month position in fiscal year (assuming calendar year)
            features['month_in_fiscal_year'] = features['month']
            features['month_remaining_in_fiscal'] = 12 - features['month']
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise

    def _create_depreciation_target(self, depreciation_history: pd.DataFrame) -> pd.Series:
        """Create target variable for depreciation prediction"""
        try:
            # Calculate next period depreciation amount
            target = depreciation_history.groupby('asset_id')['depreciation_amount'].shift(-1)
            
            return target
            
        except Exception as e:
            self.logger.error(f"Error creating depreciation target: {str(e)}")
            raise

    


    def train_model(self,
                   training_data: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train the depreciation forecasting model
        
        Args:
            training_data: DataFrame with features and target
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Separate features and target
            X = training_data[self.feature_columns]
            y = training_data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_metrics = self._calculate_regression_metrics(y_train, train_pred)
            test_metrics = self._calculate_regression_metrics(y_test, test_pred)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance()
            
            # Calculate prediction intervals using tree variance
            prediction_intervals = self._calculate_prediction_intervals(
                X_test_scaled, y_test, test_pred
            )
            
            # Store results
            self.model_metrics = {
                'training_metrics': train_metrics,
                'testing_metrics': test_metrics,
                'feature_importance': feature_importance,
                'prediction_intervals': prediction_intervals,
                'training_size': len(X_train),
                'testing_size': len(X_test),
                'training_period': self.training_period
            }
            
            return self.model_metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _calculate_regression_metrics(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict:
        """Calculate regression performance metrics"""
        try:
            metrics = {}
            
            # Calculate basic metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
            
            # Calculate custom metrics
            metrics['mean_error'] = np.mean(y_true - y_pred)
            metrics['median_error'] = np.median(y_true - y_pred)
            metrics['error_std'] = np.std(y_true - y_pred)
            
            # Calculate percentage within error bounds
            errors = np.abs(y_true - y_pred)
            metrics['within_5_percent'] = np.mean(errors <= 0.05 * y_true) * 100
            metrics['within_10_percent'] = np.mean(errors <= 0.10 * y_true) * 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {str(e)}")
            raise

    def _calculate_prediction_intervals(self,
                                     X_test: np.ndarray,
                                     y_test: np.ndarray,
                                     y_pred: np.ndarray,
                                     confidence: float = 0.95) -> Dict:
        """Calculate prediction intervals using tree variance"""
        try:
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X_test)
                for tree in self.model.estimators_
            ])
            
            # Calculate prediction intervals
            lower_bound = np.percentile(tree_predictions, (1 - confidence) * 100 / 2, axis=0)
            upper_bound = np.percentile(tree_predictions, (1 + confidence) * 100 / 2, axis=0)
            
            # Calculate coverage and width metrics
            actual_coverage = np.mean(
                (y_test >= lower_bound) & (y_test <= upper_bound)
            ) * 100
            
            interval_width = np.mean(upper_bound - lower_bound)
            relative_width = interval_width / np.mean(y_pred) * 100
            
            return {
                'confidence_level': confidence * 100,
                'actual_coverage': actual_coverage,
                'avg_interval_width': float(interval_width),
                'relative_interval_width': float(relative_width)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate and sort feature importance scores"""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Get feature importance scores
            importance_scores = self.model.feature_importances_
            
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

    def predict_depreciation(self,
                           asset_data: pd.DataFrame,
                           forecast_periods: int = 12) -> Dict:
        """
        Generate depreciation predictions for assets
        
        Args:
            asset_data: DataFrame containing asset information
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dict containing predictions and confidence intervals
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            predictions = []
            confidence_intervals = []
            
            # Generate predictions for each period
            current_date = datetime.now()
            
            for period in range(forecast_periods):
                # Prepare features for prediction
                prediction_date = current_date + timedelta(days=30 * period)
                features = self._prepare_prediction_features(asset_data, prediction_date)
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Get predictions from all trees
                tree_predictions = np.array([
                    tree.predict(features_scaled)
                    for tree in self.model.estimators_
                ])
                
                # Calculate mean prediction and confidence interval
                mean_pred = np.mean(tree_predictions, axis=0)
                lower_bound = np.percentile(tree_predictions, 5, axis=0)
                upper_bound = np.percentile(tree_predictions, 95, axis=0)
                
                predictions.append(mean_pred)
                confidence_intervals.append([lower_bound, upper_bound])
                
            # Prepare forecast results
            forecast_results = []
            for idx, row in asset_data.iterrows():
                asset_forecast = {
                    'asset_id': row['asset_id'],
                    'asset_name': row['asset_name'],
                    'forecast': {
                        'periods': list(range(1, forecast_periods + 1)),
                        'predictions': [pred[idx] for pred in predictions],
                        'confidence_intervals': [
                            {'lower': ci[0][idx], 'upper': ci[1][idx]}
                            for ci in confidence_intervals
                        ]
                    }
                }
                forecast_results.append(asset_forecast)
            
            return {
                'forecasts': forecast_results,
                'model_metrics': self.model_metrics['testing_metrics'],
                'feature_importance': dict(
                    list(self.model_metrics['feature_importance'].items())[:10]
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise



    def _prepare_prediction_features(self,
                                  asset_data: pd.DataFrame,
                                  prediction_date: datetime) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Calculate lookback period for feature creation
            lookback_start = prediction_date - timedelta(days=365)  # 1 year of history
            
            # Get historical data
            depreciation_history = self._get_depreciation_history(
                lookback_start, prediction_date
            )
            gl_data = self._get_gl_data(lookback_start, prediction_date)
            
            # Create features
            asset_features = self._create_asset_features(asset_data)
            depreciation_features = self._create_depreciation_features(depreciation_history)
            gl_features = self._create_gl_features(gl_data)
            time_features = self._create_time_features(pd.DatetimeIndex([prediction_date]))
            
            # Combine all features
            features = pd.concat([
                asset_features,
                depreciation_features.reindex(asset_data.index),
                gl_features.reindex(asset_data.index),
                time_features
            ], axis=1)
            
            # Ensure all feature columns are present in correct order
            features = features.reindex(columns=self.feature_columns, fill_value=0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            raise

    def evaluate_depreciation_patterns(self, asset_id: int) -> Dict:
        """
        Analyze depreciation patterns for a specific asset
        
        Args:
            asset_id: Asset ID to evaluate
            
        Returns:
            Dict containing depreciation pattern analysis
        """
        try:
            # Get historical data
            depreciation_history = self._get_depreciation_history(
                datetime.now() - timedelta(days=365*2),  # 2 years of history
                datetime.now()
            )
            
            # Filter for specific asset
            asset_depreciation = depreciation_history[
                depreciation_history['asset_id'] == asset_id
            ].sort_values('period_date')
            
            if asset_depreciation.empty:
                raise ValueError(f"No depreciation history found for asset {asset_id}")
            
            # Calculate pattern metrics
            pattern_metrics = {
                'avg_monthly_depreciation': float(asset_depreciation['depreciation_amount'].mean()),
                'std_monthly_depreciation': float(asset_depreciation['depreciation_amount'].std()),
                'total_depreciation': float(asset_depreciation['depreciation_amount'].sum()),
                'current_book_value': float(asset_depreciation['book_value'].iloc[-1]),
                'depreciation_rate': float(asset_depreciation['depreciation_rate'].mean() * 100)
            }
            
            # Calculate trend analysis
            trend_analysis = self._analyze_depreciation_trend(asset_depreciation)
            
            # Detect pattern anomalies
            anomalies = self._detect_depreciation_anomalies(asset_depreciation)
            
            return {
                'asset_id': asset_id,
                'pattern_metrics': pattern_metrics,
                'trend_analysis': trend_analysis,
                'anomalies': anomalies
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating depreciation patterns: {str(e)}")
            raise

    def _analyze_depreciation_trend(self,
                                  asset_depreciation: pd.DataFrame) -> Dict:
        """Analyze depreciation trends"""
        try:
            # Calculate trend components
            depreciation_values = asset_depreciation['depreciation_amount'].values
            periods = np.arange(len(depreciation_values))
            
            # Fit linear trend
            slope, intercept = np.polyfit(periods, depreciation_values, 1)
            
            # Calculate trend metrics
            trend_metrics = {
                'trend_direction': 'INCREASING' if slope > 0 else 'DECREASING',
                'trend_strength': abs(slope),
                'trend_significance': self._calculate_trend_significance(
                    depreciation_values, slope, intercept
                ),
                'seasonal_pattern': self._detect_seasonal_pattern(asset_depreciation)
            }
            
            return trend_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing depreciation trend: {str(e)}")
            raise

    def _calculate_trend_significance(self,
                                   values: np.ndarray,
                                   slope: float,
                                   intercept: float) -> str:
        """Calculate significance of trend"""
        try:
            # Calculate R-squared for trend line
            trend_line = slope * np.arange(len(values)) + intercept
            residuals = values - trend_line
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Determine significance
            if r_squared >= 0.7:
                return 'STRONG'
            elif r_squared >= 0.4:
                return 'MODERATE'
            else:
                return 'WEAK'
            
        except Exception as e:
            self.logger.error(f"Error calculating trend significance: {str(e)}")
            raise

    def _detect_seasonal_pattern(self,
                               asset_depreciation: pd.DataFrame) -> Dict:
        """Detect seasonal patterns in depreciation"""
        try:
            # Calculate monthly averages
            monthly_avg = asset_depreciation.groupby(
                asset_depreciation['period_date'].dt.month
            )['depreciation_amount'].mean()
            
            # Calculate variation coefficients
            variation_coef = monthly_avg.std() / monthly_avg.mean()
            
            # Identify peak and trough months
            peak_month = monthly_avg.idxmax()
            trough_month = monthly_avg.idxmin()
            
            return {
                'seasonal_variation': float(variation_coef),
                'has_seasonality': variation_coef > 0.1,
                'peak_month': int(peak_month),
                'trough_month': int(trough_month)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonal pattern: {str(e)}")
            raise

    def _detect_depreciation_anomalies(self,
                                     asset_depreciation: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in depreciation patterns"""
        try:
            anomalies = []
            values = asset_depreciation['depreciation_amount'].values
            dates = asset_depreciation['period_date'].values
            
            # Calculate moving average and standard deviation
            window = 6
            rolling_mean = pd.Series(values).rolling(window=window).mean()
            rolling_std = pd.Series(values).rolling(window=window).std()
            
            # Identify anomalies (outside 3 standard deviations)
            z_scores = abs((values - rolling_mean) / rolling_std)
            anomaly_indices = np.where(z_scores > 3)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'date': dates[idx].strftime('%Y-%m-%d'),
                    'value': float(values[idx]),
                    'expected_value': float(rolling_mean.iloc[idx]),
                    'deviation': float(z_scores[idx]),
                    'type': 'HIGH' if values[idx] > rolling_mean.iloc[idx] else 'LOW'
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting depreciation anomalies: {str(e)}")
            raise

    def save_model(self, model_dir: str) -> None:
        """Save trained model and associated data"""
        try:
            if self.model is None:
                raise ValueError("No trained model to save")
                
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model components
            joblib.dump(self.model, os.path.join(model_dir, 'model.joblib'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            
            # Save model info
            model_info = {
                'feature_columns': self.feature_columns,
                'training_period': self.training_period,
                'model_metrics': self.model_metrics,
                'save_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, default=str)
                
            self.logger.info(f"Model saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_dir: str) -> None:
        """Load saved model and associated data"""
        try:
            # Load model components
            self.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            
            # Load model info
            with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
                model_info = json.load(f)
                
            self.feature_columns = model_info['feature_columns']
            self.training_period = tuple(
                datetime.strptime(d, '%Y-%m-%d')
                for d in model_info['training_period']
            )
            self.model_metrics = model_info['model_metrics']
            
            self.logger.info(f"Model loaded from {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise