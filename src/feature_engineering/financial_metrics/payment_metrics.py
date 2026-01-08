# src/feature_engineering/financial_metrics/payment_metrics.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

from src.utils.database_utils import DatabaseConnector

class PaymentMetricsCalculator:
    """
    Class to calculate and analyze payment-related metrics including:
    - Payment patterns
    - Days to pay analysis
    - Supplier payment behavior
    """
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger for payment metrics"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create file handler using existing logs directory
        log_file = os.path.join('logs', 'feature_engineering', 'payment_metrics.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter matching your existing logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(handler)
        
        return logger

    def calculate_days_to_pay(self, supplier_id: Optional[int] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate days to pay metrics for invoices
        
        Args:
            supplier_id: Optional supplier ID to filter for
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            DataFrame with days to pay metrics
        """
        try:
            query = """
                SELECT 
                    i.invoice_id,
                    i.supplier_id,
                    i.invoice_date,
                    i.due_date,
                    p.payment_date,
                    i.amount as invoice_amount,
                    p.amount as payment_amount,
                    ROUND((p.payment_date - i.invoice_date)) as days_to_pay,
                    ROUND((p.payment_date - i.due_date)) as days_past_due
                FROM AP_INVOICES i
                LEFT JOIN AP_PAYMENTS p ON i.supplier_id = p.supplier_id
                WHERE i.status IN ('PAID', 'PARTIALLY_PAID')
                    {supplier_filter}
                    {date_filter}
                ORDER BY i.invoice_date
            """
            
            params = {}
            supplier_filter = ""
            date_filter = ""
            
            if supplier_id:
                supplier_filter = "AND i.supplier_id = :supplier_id"
                params['supplier_id'] = supplier_id
                
            if start_date and end_date:
                date_filter = "AND i.invoice_date BETWEEN :start_date AND :end_date"
                params['start_date'] = start_date
                params['end_date'] = end_date
                
            query = query.format(
                supplier_filter=supplier_filter,
                date_filter=date_filter
            )
            
            df = self.db.execute_query(query, params=params)
            
            if df.empty:
                return pd.DataFrame()
                
            # Calculate additional metrics
            df['on_time_payment'] = df['days_past_due'] <= 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating days to pay metrics: {str(e)}")
            raise

    def analyze_payment_patterns(self, supplier_id: Optional[int] = None,
                               months: int = 12) -> Dict[str, any]:
        """
        Analyze payment patterns and trends
        
        Args:
            supplier_id: Optional supplier ID to filter for
            months: Number of months to analyze
            
        Returns:
            Dict containing payment pattern metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30)
            
            # Get days to pay data
            days_to_pay_df = self.calculate_days_to_pay(
                supplier_id=supplier_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if days_to_pay_df.empty:
                return {
                    'avg_days_to_pay': 0,
                    'on_time_payment_rate': 0,
                    'payment_regularity': 'NO_DATA',
                    'trend': 'NO_DATA'
                }
            
            # Calculate metrics
            avg_days_to_pay = days_to_pay_df['days_to_pay'].mean()
            on_time_rate = (days_to_pay_df['on_time_payment'].sum() / 
                          len(days_to_pay_df)) * 100
            
            # Analyze payment regularity
            payment_std = days_to_pay_df['days_to_pay'].std()
            payment_regularity = self._determine_payment_regularity(payment_std)
            
            # Analyze trend
            trend = self._analyze_payment_trend(days_to_pay_df)
            
            return {
                'avg_days_to_pay': round(avg_days_to_pay, 2),
                'on_time_payment_rate': round(on_time_rate, 2),
                'payment_regularity': payment_regularity,
                'trend': trend,
                'detailed_metrics': {
                    'min_days': days_to_pay_df['days_to_pay'].min(),
                    'max_days': days_to_pay_df['days_to_pay'].max(),
                    'std_dev_days': round(payment_std, 2),
                    'total_invoices': len(days_to_pay_df),
                    'total_on_time': days_to_pay_df['on_time_payment'].sum()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing payment patterns: {str(e)}")
            raise

    def _determine_payment_regularity(self, std_dev: float) -> str:
        """Determine payment regularity based on standard deviation"""
        if std_dev <= 5:
            return 'VERY_REGULAR'
        elif std_dev <= 10:
            return 'REGULAR'
        elif std_dev <= 20:
            return 'SOMEWHAT_IRREGULAR'
        else:
            return 'IRREGULAR'

    def _analyze_payment_trend(self, df: pd.DataFrame) -> str:
        """Analyze trend in payment behavior"""
        if len(df) < 2:
            return 'INSUFFICIENT_DATA'
            
        # Calculate rolling average of days to pay
        df = df.sort_values('invoice_date')
        rolling_avg = df['days_to_pay'].rolling(window=3, min_periods=1).mean()
        
        # Compare first and last third of the data
        first_third = rolling_avg.iloc[:len(rolling_avg)//3].mean()
        last_third = rolling_avg.iloc[-len(rolling_avg)//3:].mean()
        
        diff = last_third - first_third
        
        if abs(diff) < 5:
            return 'STABLE'
        elif diff < 0:
            return 'IMPROVING'
        else:
            return 'DETERIORATING'

    def get_supplier_payment_profile(self, supplier_id: int) -> Dict[str, any]:
        """
        Get comprehensive payment profile for a supplier
        
        Args:
            supplier_id: Supplier ID to analyze
            
        Returns:
            Dict containing supplier payment profile
        """
        try:
            # Get supplier details
            supplier_query = """
                SELECT 
                    supplier_number,
                    supplier_name,
                    payment_terms,
                    status
                FROM AP_SUPPLIERS
                WHERE supplier_id = :supplier_id
            """
            
            supplier_df = self.db.execute_query(
                supplier_query, 
                params={'supplier_id': supplier_id}
            )
            
            if supplier_df.empty:
                raise ValueError(f"Supplier {supplier_id} not found")
                
            supplier_info = supplier_df.iloc[0].to_dict()
            
            # Get payment patterns
            payment_patterns = self.analyze_payment_patterns(supplier_id)
            
            # Get invoice and payment summary
            summary_query = """
                SELECT
                    COUNT(DISTINCT i.invoice_id) as total_invoices,
                    COUNT(DISTINCT p.payment_id) as total_payments,
                    SUM(i.amount) as total_invoice_amount,
                    SUM(p.amount) as total_payment_amount,
                    AVG(p.amount) as avg_payment_amount
                FROM AP_INVOICES i
                LEFT JOIN AP_PAYMENTS p ON i.supplier_id = p.supplier_id
                WHERE i.supplier_id = :supplier_id
            """
            
            summary_df = self.db.execute_query(
                summary_query,
                params={'supplier_id': supplier_id}
            )
            
            payment_summary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
            
            return {
                'supplier_info': supplier_info,
                'payment_patterns': payment_patterns,
                'payment_summary': payment_summary,
                'risk_indicators': {
                    'payment_risk_level': self._assess_payment_risk(payment_patterns),
                    'recommended_actions': self._get_recommended_actions(payment_patterns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating supplier payment profile for {supplier_id}: {str(e)}")
            raise

    def _assess_payment_risk(self, payment_patterns: Dict) -> str:
        """Assess payment risk level based on patterns"""
        if payment_patterns['avg_days_to_pay'] == 0:
            return 'UNKNOWN'
            
        risk_score = 0
        
        # Factor 1: Days to pay
        if payment_patterns['avg_days_to_pay'] <= 30:
            risk_score += 1
        elif payment_patterns['avg_days_to_pay'] <= 45:
            risk_score += 2
        else:
            risk_score += 3
            
        # Factor 2: On-time payment rate
        if payment_patterns['on_time_payment_rate'] >= 90:
            risk_score += 1
        elif payment_patterns['on_time_payment_rate'] >= 75:
            risk_score += 2
        else:
            risk_score += 3
            
        # Factor 3: Payment regularity
        if payment_patterns['payment_regularity'] in ['VERY_REGULAR', 'REGULAR']:
            risk_score += 1
        elif payment_patterns['payment_regularity'] == 'SOMEWHAT_IRREGULAR':
            risk_score += 2
        else:
            risk_score += 3
            
        # Factor 4: Trend
        if payment_patterns['trend'] == 'IMPROVING':
            risk_score -= 1
        elif payment_patterns['trend'] == 'DETERIORATING':
            risk_score += 1
            
        # Determine risk level
        if risk_score <= 4:
            return 'LOW'
        elif risk_score <= 7:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _get_recommended_actions(self, payment_patterns: Dict) -> List[str]:
        """Get recommended actions based on payment patterns"""
        actions = []
        
        if payment_patterns['avg_days_to_pay'] == 0:
            return ['Establish payment history with supplier']
            
        # Late payments
        if payment_patterns['avg_days_to_pay'] > 45:
            actions.append('Review payment terms and process for improvement opportunities')
            
        # Low on-time rate
        if payment_patterns['on_time_payment_rate'] < 75:
            actions.append('Investigate causes of late payments')
            actions.append('Consider early payment incentives')
            
        # Irregular payments
        if payment_patterns['payment_regularity'] in ['SOMEWHAT_IRREGULAR', 'IRREGULAR']:
            actions.append('Standardize payment scheduling')
            actions.append('Review invoice processing workflow')
            
        # Deteriorating trend
        if payment_patterns['trend'] == 'DETERIORATING':
            actions.append('Schedule supplier performance review')
            actions.append('Analyze root causes of payment delays')
            
        return actions if actions else ['Maintain current payment practices']