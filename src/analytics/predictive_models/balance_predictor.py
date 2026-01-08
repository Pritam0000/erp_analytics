# src/analytics/predictive_models/balance_predictor.py

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
#from src.analytics.financial_analysis.account_metrics import AccountMetricsCalculator
from src.feature_engineering.financial_metrics.account_metrics import AccountMetricsCalculator

class BalancePredictor:
    """
    Predicts account balances using historical GL data.
    Integrates with existing GL data structure and account metrics.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize predictor with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.account_metrics = AccountMetricsCalculator(data_loader)
        self.model_utils = ModelUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_period = None
        self.model_metrics = {}
        
    def prepare_training_data(self,
                            start_date: datetime,
                            end_date: datetime,
                            account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare training data for balance prediction
        
        Args:
            start_date: Start of training period
            end_date: End of training period
            account_types: Optional list of account types to include
            
        Returns:
            DataFrame with features and target
        """
        try:
            # Get required data
            gl_data = self._get_gl_data(start_date, end_date, account_types)
            account_data = self._get_account_data(account_types)
            balance_history = self._get_balance_history(start_date, end_date, account_types)
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. Account features
            account_features = self._create_account_features(account_data)
            features_df = pd.concat([features_df, account_features], axis=1)
            
            # 2. Transaction features
            transaction_features = self._create_transaction_features(gl_data)
            features_df = pd.concat([features_df, transaction_features], axis=1)
            
            # 3. Balance history features
            balance_features = self._create_balance_features(balance_history)
            features_df = pd.concat([features_df, balance_features], axis=1)
            
            # 4. Time features
            time_features = self._create_time_features(features_df.index)
            features_df = pd.concat([features_df, time_features], axis=1)
            
            # Create target variable (next period balance)
            target = self._create_balance_target(balance_history)
            
            # Combine features and target
            training_data = pd.concat([features_df, target.rename('target')], axis=1)
            training_data = training_data.dropna()
            
            # Store feature columns for later use
            self.feature_columns = features_df.columns.tolist()
            self.training_period = (start_date, end_date)
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def _get_gl_data(self,
                    start_date: datetime,
                    end_date: datetime,
                    account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get relevant GL transaction data"""
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
            
            # Get relevant transactions
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

    def _get_account_data(self, account_types: Optional[List[str]] = None) -> pd.DataFrame:
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

    def _get_balance_history(self,
                           start_date: datetime,
                           end_date: datetime,
                           account_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get historical balance data"""
        try:
            # Initialize daily balance history
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            accounts = self._get_account_data(account_types)
            
            balance_history = pd.DataFrame()
            
            # Calculate daily balances for each account
            for account_id in accounts['account_id']:
                daily_balances = pd.DataFrame(index=date_range)
                
                # Get balance for each day
                for date in date_range:
                    balance = self.account_metrics.calculate_account_balance(
                        account_id,
                        date
                    )
                    daily_balances.loc[date, 'balance'] = balance['balance']
                    
                daily_balances['account_id'] = account_id
                balance_history = pd.concat([balance_history, daily_balances])
            
            return balance_history
            
        except Exception as e:
            self.logger.error(f"Error getting balance history: {str(e)}")
            raise

    

    def _create_account_features(self, account_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from account data"""
        try:
            # Initialize features DataFrame
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
            
            # Parent account indicator
            account_features['has_parent'] = account_data['parent_account_id'].notna().astype(int)
            
            # Account level features
            account_features['account_level'] = self._calculate_account_level(account_data)
            
            # Account age features
            account_features['account_age_days'] = (
                datetime.now() - pd.to_datetime(account_data['created_date'])
            ).dt.days
            
            # Activity status
            account_features['is_active'] = (account_data['is_active'] == 'Y').astype(int)
            
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

    def _create_transaction_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from transaction data"""
        try:
            # Group by account and date
            daily_transactions = gl_data.groupby(
                ['account_id', 'journal_date']
            ).agg({
                'journal_id': 'count',
                'debit_amount': ['sum', 'count', 'mean', 'std'],
                'credit_amount': ['sum', 'count', 'mean', 'std']
            })
            
            # Flatten column names
            daily_transactions.columns = [
                'transaction_count',
                'total_debits',
                'debit_count',
                'avg_debit',
                'std_debit',
                'total_credits',
                'credit_count',
                'avg_credit',
                'std_credit'
            ]
            
            # Calculate additional metrics
            daily_transactions['net_movement'] = (
                daily_transactions['total_debits'] - daily_transactions['total_credits']
            )
            
            daily_transactions['transaction_volume'] = (
                daily_transactions['debit_count'] + daily_transactions['credit_count']
            )
            
            daily_transactions['average_transaction_size'] = (
                (daily_transactions['total_debits'] + daily_transactions['total_credits']) /
                daily_transactions['transaction_count']
            ).fillna(0)
            
            # Add rolling metrics
            for window in [7, 30, 90]:
                group = daily_transactions.groupby(level=0)
                
                daily_transactions[f'rolling_{window}d_volume'] = group['transaction_volume'].rolling(
                    window=window, min_periods=1
                ).mean()
                
                daily_transactions[f'rolling_{window}d_net'] = group['net_movement'].rolling(
                    window=window, min_periods=1
                ).mean()
                
                daily_transactions[f'rolling_{window}d_volatility'] = group['net_movement'].rolling(
                    window=window, min_periods=1
                ).std()
            
            return daily_transactions
            
        except Exception as e:
            self.logger.error(f"Error creating transaction features: {str(e)}")
            raise

    def _create_balance_features(self, balance_history: pd.DataFrame) -> pd.DataFrame:
        """Create features from balance history"""
        try:
            # Group by account and date
            balance_features = pd.DataFrame()
            
            # Calculate basic balance metrics
            balance_features['current_balance'] = balance_history['balance']
            balance_features['balance_change'] = balance_history.groupby(
                'account_id'
            )['balance'].diff()
            
            # Calculate rolling statistics
            for window in [7, 30, 90]:
                group = balance_history.groupby('account_id')
                
                # Rolling averages
                balance_features[f'balance_{window}d_avg'] = group['balance'].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling standard deviations
                balance_features[f'balance_{window}d_std'] = group['balance'].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling min/max range
                balance_features[f'balance_{window}d_range'] = (
                    group['balance'].rolling(window=window, min_periods=1).max() -
                    group['balance'].rolling(window=window, min_periods=1).min()
                )
            
            # Calculate momentum indicators
            for period in [7, 30]:
                balance_features[f'momentum_{period}d'] = (
                    balance_features['current_balance'] -
                    balance_features[f'balance_{period}d_avg']
                )
            
            # Calculate rate of change
            balance_features['daily_change_rate'] = (
                balance_features['balance_change'] /
                balance_features['current_balance'].shift(1)
            ).fillna(0)
            
            # Calculate trend indicators
            balance_features['trend_direction'] = np.sign(
                balance_features['balance_30d_avg'].diff()
            )
            
            # Add volatility metrics
            balance_features['volatility'] = balance_features['balance_30d_std'] / balance_features['balance_30d_avg']
            
            return balance_features
            
        except Exception as e:
            self.logger.error(f"Error creating balance features: {str(e)}")
            raise

    def _create_time_features(self, dates_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-based features"""
        try:
            features = pd.DataFrame(index=dates_index)
            
            # Basic time features
            features['day_of_week'] = dates_index.dayofweek
            features['day_of_month'] = dates_index.day
            features['month'] = dates_index.month
            features['quarter'] = dates_index.quarter
            features['year'] = dates_index.year
            
            # Business day indicator
            features['is_business_day'] = dates_index.dayofweek < 5
            
            # Month-end indicators
            features['is_month_end'] = dates_index.is_month_end
            features['is_quarter_end'] = dates_index.is_quarter_end
            features['is_year_end'] = dates_index.is_year_end
            
            # Days remaining in period
            features['days_in_month'] = dates_index.days_in_month
            features['days_remaining_in_month'] = (
                features['days_in_month'] - features['day_of_month']
            )
            
            # Cyclic encoding of time features
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise

    def _create_balance_target(self, balance_history: pd.DataFrame) -> pd.Series:
        """Create target variable for balance prediction"""
        try:
            # Use next day's balance as target
            target = balance_history.groupby('account_id')['balance'].shift(-1)
            return target
            
        except Exception as e:
            self.logger.error(f"Error creating balance target: {str(e)}")
            raise




    def train_model(self,
                   training_data: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train the balance prediction model
        
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
            
            # Calculate prediction intervals
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
            
            # Basic regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Calculate MAPE for non-zero values
            non_zero_mask = y_true != 0
            if non_zero_mask.any():
                metrics['mape'] = mean_absolute_percentage_error(
                    y_true[non_zero_mask],
                    y_pred[non_zero_mask]
                ) * 100
            else:
                metrics['mape'] = 0.0
            
            # Additional error metrics
            errors = y_true - y_pred
            metrics['mean_error'] = np.mean(errors)
            metrics['median_error'] = np.median(errors)
            metrics['error_std'] = np.std(errors)
            
            # Directional accuracy
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = (
                np.mean(direction_true == direction_pred) * 100
            )
            
            # Error distribution
            metrics['error_distribution'] = {
                'percentile_25': float(np.percentile(errors, 25)),
                'percentile_50': float(np.percentile(errors, 50)),
                'percentile_75': float(np.percentile(errors, 75)),
                'max_error': float(np.abs(errors).max()),
                'skewness': float(pd.Series(errors).skew()),
                'kurtosis': float(pd.Series(errors).kurtosis())
            }
            
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
            
            # Calculate intervals
            lower_percentile = (1 - confidence) * 100 / 2
            upper_percentile = 100 - lower_percentile
            
            lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
            
            # Calculate coverage statistics
            actual_coverage = np.mean(
                (y_test >= lower_bound) & (y_test <= upper_bound)
            ) * 100
            
            interval_width = upper_bound - lower_bound
            relative_width = np.mean(interval_width / np.abs(y_pred)) * 100
            
            return {
                'confidence_level': confidence * 100,
                'actual_coverage': float(actual_coverage),
                'avg_interval_width': float(np.mean(interval_width)),
                'relative_interval_width': float(relative_width),
                'width_stats': {
                    'min_width': float(np.min(interval_width)),
                    'max_width': float(np.max(interval_width)),
                    'std_width': float(np.std(interval_width))
                }
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

    def predict_balances(self,
                        account_ids: List[int],
                        forecast_horizon: int = 30) -> Dict:
        """
        Generate balance predictions for specified accounts
        
        Args:
            account_ids: List of account IDs to predict
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dict containing predictions and confidence intervals
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Get account data
            account_data = self._get_account_data()
            account_data = account_data[account_data['account_id'].isin(account_ids)]
            
            if account_data.empty:
                raise ValueError("No valid accounts found")
                
            # Initialize results
            forecast_results = []
            
            # Generate predictions for each day in the forecast horizon
            for account_id in account_ids:
                account_name = account_data[
                    account_data['account_id'] == account_id
                ]['account_name'].iloc[0]
                
                predictions = []
                confidence_intervals = []
                dates = []
                
                current_date = datetime.now()
                
                for day in range(forecast_horizon):
                    prediction_date = current_date + timedelta(days=day)
                    
                    # Prepare features
                    features = self._prepare_prediction_features(
                        account_id,
                        prediction_date
                    )
                    
                    # Scale features
                    features_scaled = self.scaler.transform(features)
                    
                    # Get predictions from all trees
                    tree_predictions = np.array([
                        tree.predict(features_scaled)
                        for tree in self.model.estimators_
                    ])
                    
                    # Calculate prediction and intervals
                    mean_pred = np.mean(tree_predictions, axis=0)[0]
                    ci = np.percentile(tree_predictions, [5, 95], axis=0).reshape(-1)
                    
                    predictions.append(mean_pred)
                    confidence_intervals.append({
                        'lower': ci[0],
                        'upper': ci[1]
                    })
                    dates.append(prediction_date.strftime('%Y-%m-%d'))
                
                forecast_results.append({
                    'account_id': account_id,
                    'account_name': account_name,
                    'forecast': {
                        'dates': dates,
                        'predictions': predictions,
                        'confidence_intervals': confidence_intervals
                    }
                })
            
            return {
                'forecasts': forecast_results,
                'model_metrics': self.model_metrics['testing_metrics'],
                'feature_importance': dict(
                    list(self.model_metrics['feature_importance'].items())[:10]
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting balances: {str(e)}")
            raise


    

    def _prepare_prediction_features(self,
                                  account_id: int,
                                  prediction_date: datetime) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Calculate lookback period for feature creation
            lookback_start = prediction_date - timedelta(days=90)  # 90 days history
            
            # Get historical data
            gl_data = self._get_gl_data(
                lookback_start,
                prediction_date,
                account_types=None  # We only need data for specific account
            )
            gl_data = gl_data[gl_data['account_id'] == account_id]
            
            account_data = self._get_account_data()
            account_data = account_data[account_data['account_id'] == account_id]
            
            balance_history = self._get_balance_history(
                lookback_start,
                prediction_date,
                account_types=None
            )
            balance_history = balance_history[balance_history['account_id'] == account_id]
            
            # Create features
            account_features = self._create_account_features(account_data)
            transaction_features = self._create_transaction_features(gl_data)
            balance_features = self._create_balance_features(balance_history)
            time_features = self._create_time_features(pd.DatetimeIndex([prediction_date]))
            
            # Combine features
            features = pd.concat([
                account_features,
                transaction_features.tail(1),
                balance_features.tail(1),
                time_features
            ], axis=1)
            
            # Ensure all feature columns are present in correct order
            features = features.reindex(columns=self.feature_columns, fill_value=0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            raise

    def analyze_balance_patterns(self, account_id: int) -> Dict:
        """
        Analyze balance patterns for a specific account
        
        Args:
            account_id: Account ID to analyze
            
        Returns:
            Dict containing balance pattern analysis
        """
        try:
            # Get historical data
            balance_history = self._get_balance_history(
                datetime.now() - timedelta(days=365),  # 1 year history
                datetime.now()
            )
            balance_history = balance_history[balance_history['account_id'] == account_id]
            
            if balance_history.empty:
                raise ValueError(f"No balance history found for account {account_id}")
            
            # Calculate pattern metrics
            pattern_metrics = {
                'avg_balance': float(balance_history['balance'].mean()),
                'std_balance': float(balance_history['balance'].std()),
                'min_balance': float(balance_history['balance'].min()),
                'max_balance': float(balance_history['balance'].max()),
                'current_balance': float(balance_history['balance'].iloc[-1])
            }
            
            # Calculate trend analysis
            trend_analysis = self._analyze_balance_trend(balance_history)
            
            # Detect pattern anomalies
            anomalies = self._detect_balance_anomalies(balance_history)
            
            # Calculate seasonality analysis
            seasonality = self._analyze_seasonality(balance_history)
            
            return {
                'account_id': account_id,
                'pattern_metrics': pattern_metrics,
                'trend_analysis': trend_analysis,
                'anomalies': anomalies,
                'seasonality': seasonality
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing balance patterns: {str(e)}")
            raise

    def _analyze_balance_trend(self, balance_history: pd.DataFrame) -> Dict:
        """Analyze balance trends"""
        try:
            # Prepare data for trend analysis
            values = balance_history['balance'].values
            dates = np.arange(len(values))
            
            # Fit linear trend
            slope, intercept = np.polyfit(dates, values, 1)
            
            # Calculate trend metrics
            trend_line = slope * dates + intercept
            residuals = values - trend_line
            
            # Calculate R-squared
            ss_tot = np.sum((values - values.mean()) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'trend_direction': 'INCREASING' if slope > 0 else 'DECREASING',
                'trend_strength': abs(slope),
                'trend_significance': 'STRONG' if r_squared > 0.7 else 'MODERATE' if r_squared > 0.4 else 'WEAK',
                'r_squared': float(r_squared),
                'monthly_growth_rate': float(slope * 30 / values.mean() * 100)  # Monthly growth as percentage
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing balance trend: {str(e)}")
            raise

    def _detect_balance_anomalies(self, balance_history: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in balance patterns"""
        try:
            anomalies = []
            values = balance_history['balance'].values
            dates = balance_history.index.values
            
            # Calculate rolling statistics
            window = 30
            rolling_mean = pd.Series(values).rolling(window=window).mean()
            rolling_std = pd.Series(values).rolling(window=window).std()
            
            # Identify anomalies (outside 3 standard deviations)
            z_scores = abs((values - rolling_mean) / rolling_std)
            anomaly_indices = np.where(z_scores > 3)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'date': pd.Timestamp(dates[idx]).strftime('%Y-%m-%d'),
                    'balance': float(values[idx]),
                    'expected_balance': float(rolling_mean.iloc[idx]),
                    'deviation': float(z_scores[idx]),
                    'type': 'HIGH' if values[idx] > rolling_mean.iloc[idx] else 'LOW'
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting balance anomalies: {str(e)}")
            raise

    def _analyze_seasonality(self, balance_history: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in balance history"""
        try:
            # Resample to monthly frequency
            monthly_balances = balance_history.resample('M')['balance'].mean()
            
            # Calculate monthly statistics
            monthly_stats = balance_history.groupby(
                balance_history.index.month
            )['balance'].agg(['mean', 'std'])
            
            # Calculate seasonal strength
            overall_std = monthly_balances.std()
            seasonal_variation = monthly_stats['std'].mean()
            seasonal_strength = seasonal_variation / overall_std if overall_std > 0 else 0
            
            # Identify peak and trough months
            peak_month = monthly_stats['mean'].idxmax()
            trough_month = monthly_stats['mean'].idxmin()
            
            return {
                'seasonal_strength': float(seasonal_strength),
                'has_seasonality': seasonal_strength > 0.1,
                'peak_month': int(peak_month),
                'trough_month': int(trough_month),
                'monthly_patterns': monthly_stats.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonality: {str(e)}")
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