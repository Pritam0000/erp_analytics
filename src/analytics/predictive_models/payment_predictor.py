# src/analytics/predictive_models/payment_predictor.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import joblib
import json
import os

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class PaymentPredictor:
    """
    Predicts payment behavior and timing using historical AP data.
    Integrates with existing AP/AR data structure.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize predictor with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.model_utils = ModelUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model attributes
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.training_period = None
        self.model_metrics = {}
        
    def prepare_training_data(self,
                            start_date: datetime,
                            end_date: datetime,
                            target_type: str = 'late_payment') -> pd.DataFrame:
        """
        Prepare training data for payment prediction
        
        Args:
            start_date: Start of training period
            end_date: End of training period
            target_type: Type of prediction ('late_payment' or 'payment_range')
            
        Returns:
            DataFrame with features and target
        """
        try:
            # Get required data
            ap_data = self._get_ap_data(start_date, end_date)
            supplier_data = self._get_supplier_data()
            payment_history = self._get_payment_history(start_date, end_date)
            
            # Create features
            features_df = pd.DataFrame()
            
            # 1. Invoice features
            invoice_features = self._create_invoice_features(ap_data)
            features_df = pd.concat([features_df, invoice_features], axis=1)
            
            # 2. Supplier features
            supplier_features = self._create_supplier_features(supplier_data, payment_history)
            features_df = pd.concat([features_df, supplier_features], axis=1)
            
            # 3. Historical payment features
            payment_features = self._create_payment_features(payment_history)
            features_df = pd.concat([features_df, payment_features], axis=1)
            
            # 4. Time features
            time_features = self._create_time_features(features_df.index)
            features_df = pd.concat([features_df, time_features], axis=1)
            
            # Create target variable based on type
            if target_type == 'late_payment':
                target = self._create_late_payment_target(ap_data)
            else:  # payment_range
                target = self._create_payment_range_target(ap_data)
                
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

    def _get_ap_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get relevant AP transaction data"""
        try:
            ap_invoices = self.data_loader.data['ap_invoices']
            ap_payments = self.data_loader.data['ap_payments']
            
            # Filter invoices for date range
            invoices = ap_invoices[
                (ap_invoices['invoice_date'].between(start_date, end_date)) &
                (ap_invoices['status'].isin(['PAID', 'PARTIALLY_PAID', 'OVERDUE']))
            ]
            
            # Join with payments data
            ap_data = pd.merge(
                invoices,
                ap_payments[['invoice_id', 'payment_date', 'payment_method', 'amount']],
                on='invoice_id',
                how='left'
            )
            
            # Calculate payment delay
            ap_data['payment_delay'] = (
                ap_data['payment_date'] - ap_data['due_date']
            ).dt.days
            
            ap_data['is_late'] = ap_data['payment_delay'] > 0
            
            return ap_data
            
        except Exception as e:
            self.logger.error(f"Error getting AP data: {str(e)}")
            raise

    def _get_supplier_data(self) -> pd.DataFrame:
        """Get supplier master data"""
        try:
            suppliers = self.data_loader.data['ap_suppliers']
            
            # Filter active suppliers
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
            
            # Filter for date range
            payment_history = ap_payments[
                ap_payments['payment_date'].between(
                    start_date - timedelta(days=180),  # Include 6 months prior
                    end_date
                )
            ].copy()
            
            return payment_history
            
        except Exception as e:
            self.logger.error(f"Error getting payment history: {str(e)}")
            raise

    

    def _create_invoice_features(self, ap_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from invoice data"""
        try:
            # Group by invoice
            invoice_features = pd.DataFrame()
            
            # Basic invoice features
            invoice_features['invoice_amount'] = ap_data['amount_x']
            invoice_features['payment_terms_days'] = (
                ap_data['due_date'] - ap_data['invoice_date']
            ).dt.days
            
            # Calculate invoice amount ranges
            amount_quantiles = ap_data['amount_x'].quantile([0.25, 0.5, 0.75])
            invoice_features['amount_quartile'] = pd.qcut(
                ap_data['amount_x'],
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
            
            # Invoice timing features
            invoice_features['month_of_year'] = ap_data['invoice_date'].dt.month
            invoice_features['day_of_month'] = ap_data['invoice_date'].dt.day
            invoice_features['day_of_week'] = ap_data['invoice_date'].dt.dayofweek
            
            # Payment deadline features
            invoice_features['due_month'] = ap_data['due_date'].dt.month
            invoice_features['due_day'] = ap_data['due_date'].dt.day
            invoice_features['days_until_due'] = (
                ap_data['due_date'] - datetime.now()
            ).dt.days
            
            # Month end indicators
            invoice_features['is_month_end_invoice'] = ap_data['invoice_date'].dt.is_month_end
            invoice_features['is_month_end_due'] = ap_data['due_date'].dt.is_month_end
            
            return invoice_features
            
        except Exception as e:
            self.logger.error(f"Error creating invoice features: {str(e)}")
            raise

    def _create_supplier_features(self,
                                supplier_data: pd.DataFrame,
                                payment_history: pd.DataFrame) -> pd.DataFrame:
        """Create features from supplier data and history"""
        try:
            supplier_features = pd.DataFrame()
            
            # Join supplier data with payment history
            supplier_history = pd.merge(
                payment_history,
                supplier_data[['supplier_id', 'payment_terms', 'status']],
                on='supplier_id'
            )
            
            # Calculate supplier payment metrics
            supplier_metrics = supplier_history.groupby('supplier_id').agg({
                'payment_id': 'count',  # Total payments
                'amount': ['sum', 'mean', 'std'],  # Payment statistics
                'payment_method': lambda x: x.value_counts().index[0]  # Most common payment method
            })
            
            # Flatten column names
            supplier_metrics.columns = [
                'total_payments',
                'total_amount',
                'avg_payment',
                'std_payment',
                'common_payment_method'
            ]
            
            # Calculate payment behavior metrics
            payment_behavior = self._calculate_payment_behavior(supplier_history)
            supplier_metrics = pd.concat([supplier_metrics, payment_behavior], axis=1)
            
            # Calculate supplier risk metrics
            risk_metrics = self._calculate_supplier_risk(supplier_history)
            supplier_metrics = pd.concat([supplier_metrics, risk_metrics], axis=1)
            
            return supplier_metrics
            
        except Exception as e:
            self.logger.error(f"Error creating supplier features: {str(e)}")
            raise

    def _calculate_payment_behavior(self, supplier_history: pd.DataFrame) -> pd.DataFrame:
        """Calculate supplier payment behavior metrics"""
        try:
            behavior_metrics = pd.DataFrame()
            
            # Group by supplier
            supplier_groups = supplier_history.groupby('supplier_id')
            
            # Calculate payment timing metrics
            payment_timing = supplier_groups.agg({
                'payment_date': lambda x: (x - supplier_history['due_date']).dt.days.mean(),  # Avg payment delay
                'amount': lambda x: (x > supplier_history['amount']).mean() * 100,  # Overpayment rate
            })
            payment_timing.columns = ['avg_payment_delay', 'overpayment_rate']
            
            # Calculate on-time payment rate
            on_time_payments = supplier_groups.apply(
                lambda x: (x['payment_date'] <= x['due_date']).mean() * 100
            )
            behavior_metrics['on_time_rate'] = on_time_payments
            
            # Calculate late payment severity
            late_severity = supplier_groups.apply(
                lambda x: np.where(
                    x['payment_date'] > x['due_date'],
                    (x['payment_date'] - x['due_date']).dt.days,
                    0
                ).mean()
            )
            behavior_metrics['late_payment_severity'] = late_severity
            
            # Combine all metrics
            behavior_metrics = pd.concat([
                behavior_metrics,
                payment_timing
            ], axis=1)
            
            return behavior_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating payment behavior: {str(e)}")
            raise

    def _calculate_supplier_risk(self, supplier_history: pd.DataFrame) -> pd.DataFrame:
        """Calculate supplier risk metrics"""
        try:
            risk_metrics = pd.DataFrame()
            
            # Group by supplier
            supplier_groups = supplier_history.groupby('supplier_id')
            
            # Calculate payment consistency
            payment_consistency = supplier_groups.agg({
                'amount': lambda x: np.std(x) / np.mean(x),  # Coefficient of variation
                'payment_date': lambda x: np.std((x - x.shift()).dt.days)  # Payment timing variability
            })
            payment_consistency.columns = ['amount_variability', 'timing_variability']
            
            # Calculate risk indicators
            risk_indicators = supplier_groups.apply(lambda x: {
                'high_risk_payments': (x['amount'] > x['amount'].mean() + 2*x['amount'].std()).mean() * 100,
                'missed_deadlines': (x['payment_date'] > x['due_date']).mean() * 100,
                'payment_gaps': (x['payment_date'].diff() > pd.Timedelta(days=45)).mean() * 100
            }).apply(pd.Series)
            
            # Combine all metrics
            risk_metrics = pd.concat([
                risk_metrics,
                payment_consistency,
                risk_indicators
            ], axis=1)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating supplier risk: {str(e)}")
            raise

    def _create_payment_features(self, payment_history: pd.DataFrame) -> pd.DataFrame:
        """Create features from payment history"""
        try:
            # Calculate payment patterns over time
            payment_features = payment_history.groupby('supplier_id').agg({
                'amount': [
                    'count',  # Number of payments
                    'sum',    # Total amount
                    'mean',   # Average amount
                    'std',    # Standard deviation
                    lambda x: (x > x.mean()).mean()  # Proportion of above-average payments
                ],
                'payment_method': lambda x: x.nunique(),  # Number of different payment methods
                'payment_date': [
                    lambda x: x.dt.dayofweek.mode()[0],  # Most common payment day
                    lambda x: x.dt.is_month_end.mean() * 100  # Percentage of month-end payments
                ]
            })
            
            # Flatten column names
            payment_features.columns = [
                'payment_count',
                'total_amount',
                'avg_amount',
                'std_amount',
                'above_avg_rate',
                'payment_methods_count',
                'common_payment_day',
                'month_end_payment_rate'
            ]
            
            # Calculate rolling averages and trends
            for window in [30, 90]:
                rolling_stats = self._calculate_rolling_payment_stats(
                    payment_history, window
                )
                payment_features = pd.concat([payment_features, rolling_stats], axis=1)
                
            return payment_features
            
        except Exception as e:
            self.logger.error(f"Error creating payment features: {str(e)}")
            raise

    def _calculate_rolling_payment_stats(self,
                                       payment_history: pd.DataFrame,
                                       window: int) -> pd.DataFrame:
        """Calculate rolling payment statistics"""
        try:
            # Sort by date
            payment_history = payment_history.sort_values('payment_date')
            
            # Calculate rolling statistics by supplier
            rolling_stats = payment_history.groupby('supplier_id').rolling(
                window=f'{window}D',
                on='payment_date',
                min_periods=1
            ).agg({
                'amount': ['mean', 'std', 'count'],
                'payment_date': lambda x: (x.max() - x.min()).days
            }).reset_index()
            
            # Rename columns
            rolling_stats.columns = [
                f'rolling_{window}d_avg_amount',
                f'rolling_{window}d_std_amount',
                f'rolling_{window}d_payment_count',
                f'rolling_{window}d_payment_span'
            ]
            
            return rolling_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling payment stats: {str(e)}")
            raise

    

    def _create_late_payment_target(self, ap_data: pd.DataFrame) -> pd.Series:
        """Create binary target for late payment prediction"""
        try:
            # Calculate payment delay
            payment_delay = (ap_data['payment_date'] - ap_data['due_date']).dt.days
            
            # Create binary target (1 for late payment, 0 for on-time)
            target = (payment_delay > 0).astype(int)
            
            return target
            
        except Exception as e:
            self.logger.error(f"Error creating late payment target: {str(e)}")
            raise

    def _create_payment_range_target(self, ap_data: pd.DataFrame) -> pd.Series:
        """Create categorical target for payment timing range"""
        try:
            # Calculate payment delay
            payment_delay = (ap_data['payment_date'] - ap_data['due_date']).dt.days
            
            # Define payment ranges
            conditions = [
                (payment_delay <= 0),                     # On time
                (payment_delay > 0) & (payment_delay <= 7),   # 1-7 days late
                (payment_delay > 7) & (payment_delay <= 30),  # 8-30 days late
                (payment_delay > 30) & (payment_delay <= 90), # 31-90 days late
                (payment_delay > 90)                          # Over 90 days late
            ]
            
            choices = ['ON_TIME', 'SLIGHTLY_LATE', 'LATE', 'VERY_LATE', 'EXTREMELY_LATE']
            
            # Create categorical target
            target = pd.Series(np.select(conditions, choices, default='UNKNOWN'))
            
            # Encode target if label encoder exists
            if self.label_encoder is not None:
                target = pd.Series(self.label_encoder.transform(target))
            
            return target
            
        except Exception as e:
            self.logger.error(f"Error creating payment range target: {str(e)}")
            raise

    def train_model(self,
                   training_data: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train the payment prediction model
        
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
            
            # Encode categorical target if needed
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=random_state
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            test_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            train_metrics = self._calculate_classification_metrics(y_train, train_pred)
            test_metrics = self._calculate_classification_metrics(y_test, test_pred)
            
            # Calculate ROC and PR curves if binary classification
            if len(np.unique(y)) == 2:
                roc_auc, pr_auc = self._calculate_curve_metrics(y_test, test_proba[:, 1])
                test_metrics.update({
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                })
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance()
            
            # Store results
            self.model_metrics = {
                'training_metrics': train_metrics,
                'testing_metrics': test_metrics,
                'feature_importance': feature_importance,
                'training_size': len(X_train),
                'testing_size': len(X_test),
                'training_period': self.training_period,
                'class_distribution': pd.Series(y).value_counts().to_dict()
            }
            
            return self.model_metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _calculate_classification_metrics(self, 
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray) -> Dict:
        """Calculate classification performance metrics"""
        try:
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Prepare metrics
            metrics = {
                'accuracy': report['accuracy'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'confusion_matrix': cm.tolist(),
                'class_metrics': {
                    label: {
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1-score']
                    }
                    for label, metrics in report.items()
                    if label not in ['accuracy', 'macro avg', 'weighted avg']
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}")
            raise

    def _calculate_curve_metrics(self,
                               y_true: np.ndarray,
                               y_prob: np.ndarray) -> Tuple[float, float]:
        """Calculate ROC and PR curve metrics"""
        try:
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Calculate PR curve and AUC
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            
            return roc_auc, pr_auc
            
        except Exception as e:
            self.logger.error(f"Error calculating curve metrics: {str(e)}")
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

    

    def predict_payment_behavior(self,
                               invoice_data: pd.DataFrame,
                               supplier_id: Optional[int] = None) -> Dict:
        """
        Predict payment behavior for new invoices
        
        Args:
            invoice_data: DataFrame containing new invoice information
            supplier_id: Optional supplier ID to filter predictions
            
        Returns:
            Dict containing predictions and confidence scores
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Filter for specific supplier if provided
            if supplier_id is not None:
                invoice_data = invoice_data[invoice_data['supplier_id'] == supplier_id]
                
            # Prepare features for prediction
            features = self._prepare_prediction_features(invoice_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Decode predictions if using label encoder
            if self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(predictions)
                
            # Prepare results
            prediction_results = []
            for idx, row in invoice_data.iterrows():
                pred_idx = len(prediction_results)
                result = {
                    'invoice_id': row.get('invoice_id', f'NEW_{idx}'),
                    'supplier_id': row['supplier_id'],
                    'invoice_amount': row['amount'],
                    'due_date': row['due_date'].strftime('%Y-%m-%d'),
                    'predicted_behavior': predictions[pred_idx],
                    'confidence_score': np.max(probabilities[pred_idx]),
                    'class_probabilities': dict(zip(
                        self.label_encoder.classes_ if self.label_encoder else range(len(probabilities[pred_idx])),
                        probabilities[pred_idx]
                    ))
                }
                prediction_results.append(result)
            
            return {
                'predictions': prediction_results,
                'model_metrics': self.model_metrics['testing_metrics'],
                'feature_importance': dict(list(self.model_metrics['feature_importance'].items())[:10])
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting payment behavior: {str(e)}")
            raise

    def _prepare_prediction_features(self, invoice_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Create invoice features
            invoice_features = self._create_invoice_features(invoice_data)
            
            # Get supplier features (using historical data)
            supplier_ids = invoice_data['supplier_id'].unique()
            historical_data = self._get_payment_history(
                datetime.now() - timedelta(days=180),
                datetime.now()
            )
            supplier_data = self._get_supplier_data()
            
            supplier_features = self._create_supplier_features(
                supplier_data[supplier_data['supplier_id'].isin(supplier_ids)],
                historical_data[historical_data['supplier_id'].isin(supplier_ids)]
            )
            
            # Create payment history features
            payment_features = self._create_payment_features(
                historical_data[historical_data['supplier_id'].isin(supplier_ids)]
            )
            
            # Create time features
            time_features = self._create_time_features(invoice_data.index)
            
            # Combine all features
            features = pd.concat([
                invoice_features,
                supplier_features.reindex(invoice_data.index),
                payment_features.reindex(invoice_data.index),
                time_features
            ], axis=1)
            
            # Ensure all feature columns are present in correct order
            features = features.reindex(columns=self.feature_columns, fill_value=0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            raise

    def evaluate_supplier_risk(self, supplier_id: int) -> Dict:
        """
        Evaluate overall payment risk for a supplier
        
        Args:
            supplier_id: Supplier ID to evaluate
            
        Returns:
            Dict containing risk assessment
        """
        try:
            # Get historical data
            historical_data = self._get_payment_history(
                datetime.now() - timedelta(days=365),
                datetime.now()
            )
            supplier_data = self._get_supplier_data()
            
            # Filter for specific supplier
            supplier_history = historical_data[historical_data['supplier_id'] == supplier_id]
            supplier_info = supplier_data[supplier_data['supplier_id'] == supplier_id]
            
            if supplier_history.empty or supplier_info.empty:
                raise ValueError(f"No data found for supplier {supplier_id}")
            
            # Calculate risk metrics
            payment_behavior = self._calculate_payment_behavior(
                pd.merge(supplier_history, supplier_info, on='supplier_id')
            )
            risk_metrics = self._calculate_supplier_risk(
                pd.merge(supplier_history, supplier_info, on='supplier_id')
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(payment_behavior, risk_metrics)
            
            return {
                'supplier_id': supplier_id,
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'payment_behavior': payment_behavior.iloc[0].to_dict(),
                'risk_metrics': risk_metrics.iloc[0].to_dict(),
                'recommendations': self._generate_risk_recommendations(risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating supplier risk: {str(e)}")
            raise

    def _calculate_risk_score(self,
                            payment_behavior: pd.DataFrame,
                            risk_metrics: pd.DataFrame) -> float:
        """Calculate composite risk score"""
        try:
            # Define weights for different components
            weights = {
                'on_time_rate': -0.3,  # Higher on-time rate reduces risk
                'late_payment_severity': 0.2,
                'amount_variability': 0.15,
                'timing_variability': 0.15,
                'high_risk_payments': 0.1,
                'missed_deadlines': 0.1
            }
            
            # Normalize metrics
            metrics = {
                'on_time_rate': payment_behavior['on_time_rate'].iloc[0] / 100,
                'late_payment_severity': min(risk_metrics['late_payment_severity'].iloc[0] / 30, 1),
                'amount_variability': min(risk_metrics['amount_variability'].iloc[0], 1),
                'timing_variability': min(risk_metrics['timing_variability'].iloc[0] / 30, 1),
                'high_risk_payments': risk_metrics['high_risk_payments'].iloc[0] / 100,
                'missed_deadlines': risk_metrics['missed_deadlines'].iloc[0] / 100
            }
            
            # Calculate weighted score
            risk_score = sum(weights[metric] * value for metric, value in metrics.items())
            
            # Convert to 0-100 scale
            risk_score = (risk_score + 1) * 50  # Convert from [-1,1] to [0,100]
            
            return max(min(risk_score, 100), 0)  # Ensure score is between 0 and 100
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            raise

    def _get_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < 20:
            return 'LOW'
        elif risk_score < 40:
            return 'LOW_MEDIUM'
        elif risk_score < 60:
            return 'MEDIUM'
        elif risk_score < 80:
            return 'MEDIUM_HIGH'
        else:
            return 'HIGH'

    def _generate_risk_recommendations(self, risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_score >= 80:
            recommendations.extend([
                "Implement strict payment monitoring",
                "Consider requiring advance payments",
                "Review credit terms immediately",
                "Establish payment schedule with milestones"
            ])
        elif risk_score >= 60:
            recommendations.extend([
                "Increase payment monitoring frequency",
                "Review and potentially adjust credit terms",
                "Consider early payment incentives",
                "Set up regular payment review meetings"
            ])
        elif risk_score >= 40:
            recommendations.extend([
                "Monitor payment patterns regularly",
                "Document any payment issues",
                "Consider early warning indicators",
                "Maintain regular communication"
            ])
        else:
            recommendations.extend([
                "Maintain current payment terms",
                "Continue regular monitoring",
                "Document good payment history"
            ])
            
        return recommendations

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
            
            if self.label_encoder is not None:
                joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))
            
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
            
            # Load label encoder if it exists
            label_encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
            
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