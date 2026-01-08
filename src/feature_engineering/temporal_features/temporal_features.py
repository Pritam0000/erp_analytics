# src/feature_engineering/temporal_features/temporal_features.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from src.utils.database_utils import DatabaseConnector

class TemporalFeatureAnalyzer:
    """
    Class for analyzing temporal features in financial data including:
    - Transaction timing patterns
    - Seasonal decomposition
    - Trend analysis
    - Frequency patterns
    - Aging analysis
    """
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger for temporal feature analysis"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = os.path.join('logs', 'feature_engineering', 'temporal_features.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
            def cleanup():
                for handler in logger.handlers:
                    handler.close()
            import atexit
            atexit.register(cleanup)
            
        return logger
    

    def analyze_transaction_timing(self, 
                               start_date: datetime,
                               end_date: datetime,
                               transaction_type: str = 'ALL') -> Dict[str, any]:
        """
        Analyze transaction timing patterns
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            transaction_type: Type of transactions to analyze ('PAYMENT', 'JOURNAL', 'ALL')
            
        Returns:
            Dict containing timing analysis results
        """
        try:
            query = """
                WITH TransactionData AS (
                    SELECT 
                        CASE
                            WHEN :transaction_type = 'PAYMENT' THEN payment_date
                            WHEN :transaction_type = 'JOURNAL' THEN journal_date
                            ELSE COALESCE(payment_date, journal_date)
                        END as transaction_date,
                        EXTRACT(HOUR FROM transaction_date) as hour_of_day,
                        EXTRACT(DOW FROM transaction_date) as day_of_week,
                        EXTRACT(DAY FROM transaction_date) as day_of_month
                    FROM (
                        SELECT journal_date, NULL as payment_date
                        FROM GL_JOURNALS
                        WHERE journal_date BETWEEN :start_date AND :end_date
                        AND :transaction_type IN ('JOURNAL', 'ALL')
                        UNION ALL
                        SELECT NULL, payment_date
                        FROM AP_PAYMENTS
                        WHERE payment_date BETWEEN :start_date AND :end_date
                        AND :transaction_type IN ('PAYMENT', 'ALL')
                    )
                )
                SELECT
                    hour_of_day,
                    day_of_week,
                    day_of_month,
                    COUNT(*) as transaction_count
                FROM TransactionData
                GROUP BY hour_of_day, day_of_week, day_of_month
            """
            
            df = self.db.execute_query(
                query,
                params={
                    'start_date': start_date,
                    'end_date': end_date,
                    'transaction_type': transaction_type
                }
            )
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    }
                }
                
            # Analyze hourly patterns
            hourly_patterns = df.groupby('hour_of_day')['transaction_count'].sum()
            peak_hour = hourly_patterns.idxmax()
            
            # Analyze daily patterns
            daily_patterns = df.groupby('day_of_week')['transaction_count'].sum()
            peak_day = daily_patterns.idxmax()
            
            # Analyze monthly patterns
            monthly_patterns = df.groupby('day_of_month')['transaction_count'].sum()
            
            return {
                'status': 'ANALYZED',
                'timing_patterns': {
                    'hourly': {
                        'peak_hour': int(peak_hour),
                        'distribution': hourly_patterns.to_dict()
                    },
                    'daily': {
                        'peak_day': int(peak_day),
                        'distribution': daily_patterns.to_dict()
                    },
                    'monthly': {
                        'distribution': monthly_patterns.to_dict()
                    }
                },
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing transaction timing: {str(e)}")
            raise

    def analyze_seasonality(self, 
                          metric_type: str,
                          start_date: datetime,
                          end_date: datetime,
                          frequency: str = 'ME') -> Dict[str, any]:
        """
        Analyze seasonal patterns in financial metrics
        
        Args:
            metric_type: Type of metric to analyze ('PAYMENTS', 'REVENUE', 'EXPENSES')
            start_date: Start of analysis period
            end_date: End of analysis period
            frequency: Seasonality frequency ('D' for daily, 'W' for weekly, 'ME' for month end)
            
        Returns:
            Dict containing seasonality analysis results
        """
        try:
            # Select appropriate metric based on type
            metric_query = {
                'PAYMENTS': """
                    SELECT payment_date as date, SUM(amount) as value
                    FROM AP_PAYMENTS
                    WHERE payment_date BETWEEN :start_date AND :end_date
                    GROUP BY payment_date
                    ORDER BY payment_date
                """,
                'REVENUE': """
                    SELECT j.journal_date as date, 
                           SUM(l.credit_amount - l.debit_amount) as value
                    FROM GL_JOURNALS j
                    JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                    JOIN GL_COA c ON l.account_id = c.account_id
                    WHERE j.journal_date BETWEEN :start_date AND :end_date
                    AND c.account_type = 'REVENUE'
                    GROUP BY j.journal_date
                    ORDER BY j.journal_date
                """,
                'EXPENSES': """
                    SELECT j.journal_date as date, 
                           SUM(l.debit_amount - l.credit_amount) as value
                    FROM GL_JOURNALS j
                    JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                    JOIN GL_COA c ON l.account_id = c.account_id
                    WHERE j.journal_date BETWEEN :start_date AND :end_date
                    AND c.account_type = 'EXPENSE'
                    GROUP BY j.journal_date
                    ORDER BY j.journal_date
                """
            }
            
            if metric_type not in metric_query:
                raise ValueError(f"Invalid metric type: {metric_type}")
                
            df = self.db.execute_query(
                metric_query[metric_type],
                params={
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'metric_type': metric_type
                }
            
            try:
                # Convert date column to datetime if it's not already
                df['date'] = pd.to_datetime(df['date'])
                # Set date as index and resample to desired frequency
                df.set_index('date', inplace=True)
                df = df.resample(frequency).sum()
                
                # Ensure we have enough periods for decomposition
                if len(df) < 24:  # Need at least 2 complete cycles
                    return {
                        'status': 'INSUFFICIENT_DATA',
                        'metric_type': metric_type,
                        'required_periods': 24,
                        'available_periods': len(df)
                    }
                
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    df['value'],
                    period=12 if frequency == 'ME' else 52 if frequency == 'W' else 7
                )
                
                # Calculate seasonal strength
                seasonal_strength = 1 - (np.var(decomposition.resid) / 
                                      np.var(decomposition.seasonal + decomposition.resid))
                
                # Identify peak and trough periods
                seasonal_pattern = pd.DataFrame({
                    'seasonal': decomposition.seasonal
                })
                peak_period = seasonal_pattern['seasonal'].idxmax()
                trough_period = seasonal_pattern['seasonal'].idxmin()
                
                return {
                    'status': 'ANALYZED',
                    'seasonality_metrics': {
                        'seasonal_strength': float(seasonal_strength),
                        'peak_period': peak_period.strftime('%Y-%m-%d'),
                        'trough_period': trough_period.strftime('%Y-%m-%d'),
                        'components': {
                            'trend': decomposition.trend.tolist(),
                            'seasonal': decomposition.seasonal.tolist(),
                            'residual': decomposition.resid.tolist()
                        }
                    },
                    'period': {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'frequency': frequency
                    }
                }
            except Exception as inner_e:
                self.logger.error(f"Error processing seasonal data: {str(inner_e)}")
                raise ValueError(f"Error processing seasonal data: {str(inner_e)}")
            
        except Exception as e:
            self.logger.error(f"Error in seasonality analysis: {str(e)}")
            raise

    def analyze_aging_metrics(self,
                            metric_type: str,
                            as_of_date: Optional[datetime] = None) -> Dict[str, any]:
        """
        Analyze aging metrics for various financial items
        
        Args:
            metric_type: Type of aging to analyze ('RECEIVABLES', 'PAYABLES', 'INVENTORY')
            as_of_date: Date for aging analysis, defaults to current date
            
        Returns:
            Dict containing aging analysis results
        """
        try:
            as_of_date = as_of_date or datetime.now()
            
            aging_queries = {
                'RECEIVABLES': """
                    SELECT 
                        CASE 
                            WHEN days_outstanding <= 30 THEN '0-30'
                            WHEN days_outstanding <= 60 THEN '31-60'
                            WHEN days_outstanding <= 90 THEN '61-90'
                            ELSE 'Over 90'
                        END as aging_bucket,
                        COUNT(*) as item_count,
                        SUM(remaining_amount) as total_amount
                    FROM (
                        SELECT 
                            i.invoice_id,
                            i.amount - COALESCE(SUM(p.amount), 0) as remaining_amount,
                            EXTRACT(DAY FROM :as_of_date - i.invoice_date) as days_outstanding
                        FROM AP_INVOICES i
                        LEFT JOIN AP_PAYMENTS p ON i.invoice_id = p.invoice_id
                        WHERE i.status != 'PAID'
                        GROUP BY i.invoice_id, i.amount, i.invoice_date
                    )
                    GROUP BY aging_bucket
                """,
                'PAYABLES': """
                    SELECT 
                        CASE 
                            WHEN days_outstanding <= 30 THEN '0-30'
                            WHEN days_outstanding <= 60 THEN '31-60'
                            WHEN days_outstanding <= 90 THEN '61-90'
                            ELSE 'Over 90'
                        END as aging_bucket,
                        COUNT(*) as item_count,
                        SUM(remaining_amount) as total_amount
                    FROM (
                        SELECT 
                            i.invoice_id,
                            i.amount - COALESCE(SUM(p.amount), 0) as remaining_amount,
                            EXTRACT(DAY FROM :as_of_date - i.due_date) as days_outstanding
                        FROM AP_INVOICES i
                        LEFT JOIN AP_PAYMENTS p ON i.invoice_id = p.invoice_id
                        WHERE i.status != 'PAID'
                        GROUP BY i.invoice_id, i.amount, i.due_date
                    )
                    GROUP BY aging_bucket
                """
            }
            
            if metric_type not in aging_queries:
                raise ValueError(f"Invalid aging metric type: {metric_type}")
                
            df = self.db.execute_query(
                aging_queries[metric_type],
                params={'as_of_date': as_of_date}
            )
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'metric_type': metric_type,
                    'as_of_date': as_of_date
                }
                
            # Calculate aging statistics
            total_amount = df['total_amount'].sum()
            total_items = df['item_count'].sum()
            
            aging_stats = df.set_index('aging_bucket').to_dict('index')
            
            # Add percentage calculations
            for bucket, stats in aging_stats.items():
                stats['amount_percentage'] = (stats['total_amount'] / total_amount * 100 
                                           if total_amount > 0 else 0)
                stats['item_percentage'] = (stats['item_count'] / total_items * 100 
                                         if total_items > 0 else 0)
            
            return {
                'status': 'ANALYZED',
                'aging_metrics': {
                    'total_amount': float(total_amount),
                    'total_items': int(total_items),
                    'aging_buckets': aging_stats,
                    'risk_indicators': self._calculate_aging_risk(aging_stats)
                },
                'as_of_date': as_of_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            self.logger.error(f"Error in aging analysis: {str(e)}")
            raise
            
    def _calculate_aging_risk(self, aging_stats: Dict) -> Dict[str, any]:
        """Calculate risk indicators based on aging statistics"""
        try:
            # Calculate weighted average age
            total_weighted_age = 0
            total_amount = 0
            
            age_weights = {
                '0-30': 15,
                '31-60': 45,
                '61-90': 75,
                'Over 90': 105
            }
            
            for bucket, stats in aging_stats.items():
                total_weighted_age += age_weights[bucket] * stats['total_amount']
                total_amount += stats['total_amount']
                
            avg_age = total_weighted_age / total_amount if total_amount > 0 else 0
            
            # Calculate risk level
            if avg_age <= 30:
                risk_level = 'LOW'
            elif avg_age <= 60:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
                
            # Calculate concentration in older buckets
            old_bucket_concentration = (
                aging_stats.get('Over 90', {}).get('amount_percentage', 0)
            )
            
            return {
                'average_age': round(avg_age, 2),
                'risk_level': risk_level,
                'old_bucket_concentration': round(old_bucket_concentration, 2),
                'recommended_actions': self._get_aging_recommendations(
                    risk_level,
                    old_bucket_concentration
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating aging risk: {str(e)}")
            raise
            
    def _get_aging_recommendations(self, risk_level: str, 
                                 old_concentration: float) -> List[str]:
        """Get recommendations based on aging risk analysis"""
        recommendations = []
        
        if risk_level == 'HIGH':
            recommendations.extend([
                "Review collection/payment strategies",
                "Consider early payment incentives",
                "Implement stricter credit controls"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Monitor aging trends closely",
                "Review payment terms with key accounts"
            ])
            
        if old_concentration > 20:
            recommendations.extend([
                "Focus on collecting aged balances",
                "Review credit limits for accounts with old balances"
            ])
            
        return recommendations or ["Maintain current practices"]