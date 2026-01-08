# src/analytics/predictive_models/cash_flow_forecaster.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader
from src.analytics.financial_analysis.cash_flow_analyzer import CashFlowAnalyzer

class CashFlowForecaster:
    """
    Predicts future cash flows using historical GL and AP/AR data.
    Integrates with our financial data structure and cash flow analysis.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize forecaster with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.model_utils = ModelUtils()
        self.cash_flow_analyzer = CashFlowAnalyzer(data_loader)
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_period = None
        self.forecast_horizon = None
        self.model_metrics = {}
        
    def prepare_training_data(self,
                            start_date: datetime,
                            end_date: datetime,
                            target_type: str = 'operating') -> pd.DataFrame:
        """
        Prepare training data for cash flow forecasting
        
        Args:
            start_date: Start of training period
            end_date: End of training period
            target_type: Type of cash flow to predict ('operating', 'investing', 'financing', 'net')
            
        Returns:
            DataFrame with features and target
        """
        try:
            # Get required data
            gl_data = self._get_gl_data(start_date, end_date)
            ap_data = self._get_ap_data(start_date, end_date)
            historical_cash_flows = self._get_historical_cash_flows(start_date, end_date)
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. GL-based features
            gl_features = self._create_gl_features(gl_data)
            features_df = pd.concat([features_df, gl_features], axis=1)
            
            # 2. AP/AR features
            ap_features = self._create_ap_features(ap_data)
            features_df = pd.concat([features_df, ap_features], axis=1)
            
            # 3. Historical cash flow features
            cf_features = self._create_cash_flow_features(historical_cash_flows)
            features_df = pd.concat([features_df, cf_features], axis=1)
            
            # 4. Time-based features
            time_features = self._create_time_features(features_df.index)
            features_df = pd.concat([features_df, time_features], axis=1)
            
            # Create target variable
            if target_type == 'operating':
                target = historical_cash_flows['operating_cash_flow']
            elif target_type == 'investing':
                target = historical_cash_flows['investing_cash_flow']
            elif target_type == 'financing':
                target = historical_cash_flows['financing_cash_flow']
            else:  # net cash flow
                target = historical_cash_flows['net_cash_flow']
            
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
            
    def _get_gl_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get relevant GL transaction data"""
        try:
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            
            # Filter for date range
            gl_journals = gl_journals[
                (gl_journals['journal_date'].between(start_date, end_date)) &
                (gl_journals['status'] == 'POSTED')
            ]
            
            # Join with journal lines
            gl_data = gl_lines[
                gl_lines['journal_id'].isin(gl_journals['journal_id'])
            ].merge(
                gl_journals[['journal_id', 'journal_date']],
                on='journal_id'
            )
            
            return gl_data
            
        except Exception as e:
            self.logger.error(f"Error getting GL data: {str(e)}")
            raise
            
    def _get_ap_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get relevant AP transaction data"""
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
            
            # Combine into single DataFrame with payment status
            ap_data = pd.merge(
                invoices,
                payments[['invoice_id', 'payment_date', 'amount']],
                on='invoice_id',
                how='left'
            )
            
            return ap_data
            
        except Exception as e:
            self.logger.error(f"Error getting AP data: {str(e)}")
            raise

    

    def _get_historical_cash_flows(self, 
                                 start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """Get historical cash flow data"""
        try:
            # Initialize empty DataFrame for cash flows
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            cash_flows = pd.DataFrame(index=dates)
            
            # Iterate through dates to get daily cash flows
            for date in dates:
                cf_data = self.cash_flow_analyzer.analyze_cash_flow(
                    start_date=date,
                    end_date=date
                )
                
                cash_flows.loc[date, 'operating_cash_flow'] = cf_data['cash_flows']['operating']['net_cash']
                cash_flows.loc[date, 'investing_cash_flow'] = cf_data['cash_flows']['investing']['net_cash']
                cash_flows.loc[date, 'financing_cash_flow'] = cf_data['cash_flows']['financing']['net_cash']
                cash_flows.loc[date, 'net_cash_flow'] = cf_data['cash_flows']['net_cash_flow']
            
            return cash_flows
            
        except Exception as e:
            self.logger.error(f"Error getting historical cash flows: {str(e)}")
            raise

    def _create_gl_features(self, gl_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from GL transactions"""
        try:
            # Group by date
            daily_gl = gl_data.groupby('journal_date').agg({
                'journal_id': 'count',  # Number of journals
                'debit_amount': ['sum', 'mean', 'std'],  # Debit statistics
                'credit_amount': ['sum', 'mean', 'std'],  # Credit statistics
                'account_id': 'nunique'  # Number of unique accounts
            })
            
            # Flatten column names
            daily_gl.columns = [
                'journal_count',
                'total_debits',
                'avg_debit',
                'std_debit',
                'total_credits',
                'avg_credit',
                'std_credit',
                'unique_accounts'
            ]
            
            # Calculate additional features
            daily_gl['net_position'] = daily_gl['total_debits'] - daily_gl['total_credits']
            daily_gl['transaction_volume'] = daily_gl['total_debits'] + daily_gl['total_credits']
            daily_gl['avg_transaction_size'] = (
                daily_gl['transaction_volume'] / daily_gl['journal_count']
            )
            
            # Add rolling averages
            for window in [3, 7, 30]:
                daily_gl[f'net_position_{window}d_avg'] = daily_gl['net_position'].rolling(window).mean()
                daily_gl[f'volume_{window}d_avg'] = daily_gl['transaction_volume'].rolling(window).mean()
                
            return daily_gl
            
        except Exception as e:
            self.logger.error(f"Error creating GL features: {str(e)}")
            raise

    def _create_ap_features(self, ap_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from AP transactions"""
        try:
            # Calculate days to pay
            ap_data['days_to_pay'] = (
                ap_data['payment_date'] - ap_data['invoice_date']
            ).dt.days
            
            # Create daily features
            daily_ap = ap_data.groupby('invoice_date').agg({
                'invoice_id': 'count',  # Number of invoices
                'amount_x': ['sum', 'mean', 'std'],  # Invoice amounts
                'amount_y': 'sum',  # Payment amounts
                'days_to_pay': ['mean', 'std']  # Payment timing
            })
            
            # Flatten column names
            daily_ap.columns = [
                'invoice_count',
                'total_invoiced',
                'avg_invoice',
                'std_invoice',
                'total_paid',
                'avg_days_to_pay',
                'std_days_to_pay'
            ]
            
            # Calculate additional features
            daily_ap['unpaid_amount'] = daily_ap['total_invoiced'] - daily_ap['total_paid']
            daily_ap['payment_rate'] = (
                daily_ap['total_paid'] / daily_ap['total_invoiced']
            ).fillna(0)
            
            # Add rolling averages
            for window in [7, 30, 90]:
                daily_ap[f'unpaid_{window}d_avg'] = daily_ap['unpaid_amount'].rolling(window).mean()
                daily_ap[f'payment_rate_{window}d_avg'] = daily_ap['payment_rate'].rolling(window).mean()
                
            return daily_ap
            
        except Exception as e:
            self.logger.error(f"Error creating AP features: {str(e)}")
            raise

    def _create_cash_flow_features(self, historical_cash_flows: pd.DataFrame) -> pd.DataFrame:
        """Create features from historical cash flows"""
        try:
            features = pd.DataFrame(index=historical_cash_flows.index)
            
            # Create lagged features
            for lag in [1, 3, 7, 14, 30]:
                for col in historical_cash_flows.columns:
                    features[f'{col}_lag_{lag}'] = historical_cash_flows[col].shift(lag)
            
            # Calculate rolling statistics
            for window in [7, 30, 90]:
                for col in historical_cash_flows.columns:
                    # Rolling means
                    features[f'{col}_{window}d_avg'] = (
                        historical_cash_flows[col].rolling(window).mean()
                    )
                    # Rolling standard deviations
                    features[f'{col}_{window}d_std'] = (
                        historical_cash_flows[col].rolling(window).std()
                    )
                    # Rolling min/max range
                    features[f'{col}_{window}d_range'] = (
                        historical_cash_flows[col].rolling(window).max() -
                        historical_cash_flows[col].rolling(window).min()
                    )
            
            # Calculate momentum indicators
            for period in [7, 30]:
                for col in historical_cash_flows.columns:
                    features[f'{col}_momentum_{period}d'] = (
                        historical_cash_flows[col] -
                        historical_cash_flows[col].shift(period)
                    )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating cash flow features: {str(e)}")
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
            
            # Cyclic encoding of time features
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Is business day feature
            features['is_business_day'] = dates_index.dayofweek < 5
            
            # End of month/quarter indicators
            features['is_month_end'] = dates_index.is_month_end
            features['is_quarter_end'] = dates_index.is_quarter_end
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise

    

    def train_model(self,
                   training_data: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train the cash flow forecasting model
        
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
            
            # Split data into training and testing sets
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
            train_metrics = self._calculate_model_metrics(y_train, train_pred)
            test_metrics = self._calculate_model_metrics(y_test, test_pred)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance()
            
            # Store results
            self.model_metrics = {
                'training_metrics': train_metrics,
                'testing_metrics': test_metrics,
                'feature_importance': feature_importance,
                'training_size': len(X_train),
                'testing_size': len(X_test),
                'training_period': self.training_period
            }
            
            return self.model_metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model performance metrics"""
        try:
            metrics = {}
            
            # Calculate basic metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Calculate mean absolute percentage error (MAPE)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape
            
            # Calculate directional accuracy
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(direction_true == direction_pred) * 100
            metrics['directional_accuracy'] = directional_accuracy
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate and sort feature importance scores"""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            importance_scores = self.model.feature_importances_
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

    def predict_cash_flow(self,
                         prediction_date: datetime,
                         forecast_horizon: int = 30) -> Dict:
        """
        Generate cash flow predictions
        
        Args:
            prediction_date: Date to start predictions from
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dict containing predictions and confidence intervals
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Generate dates for prediction period
            prediction_dates = pd.date_range(
                start=prediction_date,
                periods=forecast_horizon,
                freq='D'
            )
            
            predictions = []
            confidence_intervals = []
            
            # Generate predictions for each date
            for date in prediction_dates:
                # Prepare features for prediction
                features = self._prepare_prediction_features(date)
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction with all trees
                tree_predictions = np.array([
                    tree.predict(features_scaled)
                    for tree in self.model.estimators_
                ])
                
                # Calculate mean prediction and confidence interval
                mean_pred = np.mean(tree_predictions, axis=0)[0]
                confidence_interval = np.percentile(tree_predictions, [5, 95], axis=0).reshape(-1)
                
                predictions.append(mean_pred)
                confidence_intervals.append(confidence_interval)
            
            # Prepare results
            forecast_results = {
                'dates': prediction_dates,
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'forecast_horizon': forecast_horizon,
                'prediction_date': prediction_date,
                'model_metrics': self.model_metrics
            }
            
            return forecast_results
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise

    def _prepare_prediction_features(self, prediction_date: datetime) -> pd.DataFrame:
        """Prepare features for a specific prediction date"""
        try:
            # Calculate lookback period for feature creation
            lookback_start = prediction_date - timedelta(days=90)  # 90 days for sufficient history
            
            # Get historical data
            gl_data = self._get_gl_data(lookback_start, prediction_date)
            ap_data = self._get_ap_data(lookback_start, prediction_date)
            historical_cash_flows = self._get_historical_cash_flows(lookback_start, prediction_date)
            
            # Create features
            gl_features = self._create_gl_features(gl_data)
            ap_features = self._create_ap_features(ap_data)
            cf_features = self._create_cash_flow_features(historical_cash_flows)
            time_features = self._create_time_features(pd.DatetimeIndex([prediction_date]))
            
            # Combine all features
            features = pd.concat([
                gl_features.iloc[-1:],
                ap_features.iloc[-1:],
                cf_features.iloc[-1:],
                time_features
            ], axis=1)
            
            # Ensure all feature columns are present in correct order
            features = features.reindex(columns=self.feature_columns, fill_value=0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            raise

    

    def evaluate_model(self, evaluation_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on new data
        
        Args:
            evaluation_data: DataFrame with features and actual values
            
        Returns:
            Dict containing evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Separate features and target
            X_eval = evaluation_data[self.feature_columns]
            y_eval = evaluation_data['target']
            
            # Scale features
            X_eval_scaled = self.scaler.transform(X_eval)
            
            # Generate predictions
            predictions = self.model.predict(X_eval_scaled)
            
            # Calculate metrics
            eval_metrics = self._calculate_model_metrics(y_eval, predictions)
            
            # Calculate prediction errors
            errors = y_eval - predictions
            
            # Additional error analysis
            eval_metrics.update({
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(np.abs(errors)),
                'evaluation_size': len(X_eval)
            })
            
            return eval_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, model_dir: str) -> None:
        """
        Save trained model and associated data
        
        Args:
            model_dir: Directory to save model files
        """
        try:
            if self.model is None:
                raise ValueError("No trained model to save")
                
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model components
            joblib.dump(self.model, os.path.join(model_dir, 'model.joblib'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            
            # Save feature columns and metrics
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
        """
        Load saved model and associated data
        
        Args:
            model_dir: Directory containing model files
        """
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

    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return {
            'model_info': {
                'model_type': type(self.model).__name__,
                'n_features': len(self.feature_columns),
                'training_period': {
                    'start_date': self.training_period[0].strftime('%Y-%m-%d'),
                    'end_date': self.training_period[1].strftime('%Y-%m-%d')
                }
            },
            'performance_metrics': {
                'training': self.model_metrics['training_metrics'],
                'testing': self.model_metrics['testing_metrics']
            },
            'top_features': dict(
                list(self.model_metrics['feature_importance'].items())[:10]
            ),
            'training_data_info': {
                'training_samples': self.model_metrics['training_size'],
                'testing_samples': self.model_metrics['testing_size']
            }
        }

    def analyze_prediction_errors(self, 
                                actual_values: pd.Series,
                                predictions: pd.Series) -> Dict:
        """
        Analyze prediction errors in detail
        
        Args:
            actual_values: Series of actual values
            predictions: Series of predicted values
            
        Returns:
            Dict containing error analysis
        """
        try:
            errors = actual_values - predictions
            percentage_errors = (errors / actual_values) * 100
            
            # Basic error statistics
            error_stats = {
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'mean_absolute_error': np.mean(np.abs(errors)),
                'mean_percentage_error': np.mean(percentage_errors),
                'median_percentage_error': np.median(percentage_errors)
            }
            
            # Error distribution analysis
            error_distribution = {
                'percentiles': {
                    '5th': np.percentile(errors, 5),
                    '25th': np.percentile(errors, 25),
                    '75th': np.percentile(errors, 75),
                    '95th': np.percentile(errors, 95)
                },
                'skewness': float(pd.Series(errors).skew()),
                'kurtosis': float(pd.Series(errors).kurtosis())
            }
            
            # Error classification
            error_classification = {
                'large_errors': np.sum(np.abs(percentage_errors) > 20),
                'overestimates': np.sum(errors < 0),
                'underestimates': np.sum(errors > 0)
            }
            
            # Combine results
            error_analysis = {
                'error_statistics': error_stats,
                'error_distribution': error_distribution,
                'error_classification': error_classification,
                'sample_size': len(actual_values)
            }
            
            return error_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction errors: {str(e)}")
            raise

    def plot_forecasts(self,
                      forecast_results: Dict,
                      actual_values: Optional[pd.Series] = None) -> None:
        """
        Plot forecasts with confidence intervals
        
        Args:
            forecast_results: Results from predict_cash_flow
            actual_values: Optional series of actual values for comparison
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot predictions with confidence intervals
            dates = forecast_results['dates']
            predictions = forecast_results['predictions']
            intervals = np.array(forecast_results['confidence_intervals'])
            
            plt.plot(dates, predictions, 'b-', label='Predicted')
            plt.fill_between(dates,
                           intervals[:, 0],
                           intervals[:, 1],
                           alpha=0.2,
                           label='90% Confidence Interval')
                           
            # Add actual values if provided
            if actual_values is not None:
                plt.plot(dates, actual_values, 'r--', label='Actual')
                
            plt.title('Cash Flow Forecast')
            plt.xlabel('Date')
            plt.ylabel('Cash Flow')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting forecasts: {str(e)}")
            raise

    def get_forecast_summary(self, forecast_results: Dict) -> Dict:
        """Generate summary of forecast results"""
        try:
            summary = {
                'forecast_period': {
                    'start_date': forecast_results['prediction_date'].strftime('%Y-%m-%d'),
                    'horizon': forecast_results['forecast_horizon']
                },
                'forecast_statistics': {
                    'mean_forecast': np.mean(forecast_results['predictions']),
                    'min_forecast': np.min(forecast_results['predictions']),
                    'max_forecast': np.max(forecast_results['predictions']),
                    'std_forecast': np.std(forecast_results['predictions'])
                },
                'confidence_intervals': {
                    'lower_bound': np.mean([ci[0] for ci in forecast_results['confidence_intervals']]),
                    'upper_bound': np.mean([ci[1] for ci in forecast_results['confidence_intervals']]),
                    'average_range': np.mean([ci[1] - ci[0] for ci in forecast_results['confidence_intervals']])
                },
                'model_performance': {
                    'r2_score': self.model_metrics['testing_metrics']['r2'],
                    'mape': self.model_metrics['testing_metrics']['mape'],
                    'directional_accuracy': self.model_metrics['testing_metrics']['directional_accuracy']
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating forecast summary: {str(e)}")
            raise