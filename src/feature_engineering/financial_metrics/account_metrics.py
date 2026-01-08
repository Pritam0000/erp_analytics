# src/feature_engineering/financial_metrics/account_metrics.py
import os
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

from src.utils.database_utils import DatabaseConnector
from src.data_loading import FinancialDataLoader

class AccountMetricsCalculator:
    """
    Class to calculate and analyze GL account metrics including:
    - Account balances
    - Activity patterns
    - Balance trends
    """
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger for account metrics"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create file handler using existing logs directory
        log_file = os.path.join('logs', 'feature_engineering', 'account_metrics.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter matching your existing logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(handler)
        
        return logger

    def calculate_account_balance(self, account_id: int, as_of_date: datetime) -> Dict[str, float]:
        """
        Calculate account balance and related metrics as of a specific date
        
        Args:
            account_id: The GL account ID
            as_of_date: The date to calculate balance for
            
        Returns:
            Dict containing balance metrics
        """
        try:
            query = """
                SELECT 
                    COALESCE(SUM(debit_amount), 0) as total_debits,
                    COALESCE(SUM(credit_amount), 0) as total_credits
                FROM GL_JOURNAL_LINES gl
                JOIN GL_JOURNALS j ON gl.journal_id = j.journal_id
                WHERE gl.account_id = :account_id
                AND j.journal_date <= :as_of_date
                AND j.status = 'POSTED'
            """
            
            df = self.db.execute_query(
                query, 
                params={'account_id': account_id, 'as_of_date': as_of_date}
            )
            
            if df.empty:
                return {'balance': 0.0, 'total_debits': 0.0, 'total_credits': 0.0}
            
            row = df.iloc[0]
            balance = row['total_debits'] - row['total_credits']
            
            return {
                'balance': balance,
                'total_debits': row['total_debits'],
                'total_credits': row['total_credits']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating balance for account {account_id}: {str(e)}")
            raise

    def get_account_activity_metrics(self, account_id: int, 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, any]:
        """
        Calculate account activity metrics for a given period
        
        Args:
            account_id: The GL account ID
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dict containing activity metrics
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as transaction_count,
                    COUNT(DISTINCT journal_id) as journal_count,
                    MAX(journal_date) as last_activity_date,
                    COALESCE(SUM(CASE WHEN debit_amount > 0 THEN 1 ELSE 0 END), 0) as debit_count,
                    COALESCE(SUM(CASE WHEN credit_amount > 0 THEN 1 ELSE 0 END), 0) as credit_count,
                    COALESCE(MAX(debit_amount), 0) as max_debit,
                    COALESCE(MAX(credit_amount), 0) as max_credit
                FROM GL_JOURNAL_LINES gl
                JOIN GL_JOURNALS j ON gl.journal_id = j.journal_id
                WHERE gl.account_id = :account_id
                AND j.journal_date BETWEEN :start_date AND :end_date
                AND j.status = 'POSTED'
            """
            
            df = self.db.execute_query(
                query,
                params={
                    'account_id': account_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            if df.empty:
                return {
                    'transaction_count': 0,
                    'journal_count': 0,
                    'last_activity_date': None,
                    'debit_count': 0,
                    'credit_count': 0,
                    'max_debit': 0.0,
                    'max_credit': 0.0,
                    'activity_level': 'INACTIVE'
                }
            
            row = df.iloc[0]
            
            # Calculate activity level
            monthly_transaction_avg = row['transaction_count'] / ((end_date - start_date).days / 30)
            activity_level = self._determine_activity_level(monthly_transaction_avg)
            
            return {
                'transaction_count': row['transaction_count'],
                'journal_count': row['journal_count'],
                'last_activity_date': row['last_activity_date'],
                'debit_count': row['debit_count'],
                'credit_count': row['credit_count'],
                'max_debit': row['max_debit'],
                'max_credit': row['max_credit'],
                'activity_level': activity_level
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating activity metrics for account {account_id}: {str(e)}")
            raise

    def _determine_activity_level(self, monthly_transactions: float) -> str:
        """Determine account activity level based on monthly transaction volume"""
        if monthly_transactions >= 100:
            return 'VERY_HIGH'
        elif monthly_transactions >= 50:
            return 'HIGH'
        elif monthly_transactions >= 10:
            return 'MEDIUM'
        elif monthly_transactions > 0:
            return 'LOW'
        return 'INACTIVE'

    def calculate_balance_trends(self, account_id: int, 
                               months: int = 12) -> pd.DataFrame:
        """
        Calculate monthly balance trends for an account
        
        Args:
            account_id: The GL account ID
            months: Number of months to analyze
            
        Returns:
            DataFrame with monthly balance trends
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30)
            
            query = """
                WITH RECURSIVE dates AS (
                    SELECT TRUNC(:start_date, 'MM') as month_start
                    UNION ALL
                    SELECT ADD_MONTHS(month_start, 1)
                    FROM dates
                    WHERE month_start < TRUNC(:end_date, 'MM')
                )
                SELECT 
                    d.month_start,
                    COALESCE(SUM(gl.debit_amount), 0) as monthly_debits,
                    COALESCE(SUM(gl.credit_amount), 0) as monthly_credits,
                    COUNT(DISTINCT gl.journal_id) as transaction_count
                FROM dates d
                LEFT JOIN GL_JOURNALS j ON TRUNC(j.journal_date, 'MM') = d.month_start
                    AND j.status = 'POSTED'
                LEFT JOIN GL_JOURNAL_LINES gl ON gl.journal_id = j.journal_id
                    AND gl.account_id = :account_id
                GROUP BY d.month_start
                ORDER BY d.month_start
            """
            
            df = self.db.execute_query(
                query,
                params={
                    'account_id': account_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Calculate running balance
            df['net_change'] = df['monthly_debits'] - df['monthly_credits']
            df['running_balance'] = df['net_change'].cumsum()
            
            # Calculate month-over-month changes
            df['balance_change_pct'] = df['running_balance'].pct_change() * 100
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating balance trends for account {account_id}: {str(e)}")
            raise

    def get_account_summary(self, account_id: int) -> Dict[str, any]:
        """
        Get comprehensive account summary including current metrics and trends
        
        Args:
            account_id: The GL account ID
            
        Returns:
            Dict containing account summary metrics
        """
        try:
            # Get account details
            account_query = """
                SELECT 
                    account_number,
                    account_name,
                    account_type,
                    account_category,
                    is_active
                FROM GL_COA
                WHERE account_id = :account_id
            """
            
            account_df = self.db.execute_query(account_query, params={'account_id': account_id})
            if account_df.empty:
                raise ValueError(f"Account {account_id} not found")
            
            account_info = account_df.iloc[0].to_dict()
            
            # Get current balance
            current_balance = self.calculate_account_balance(
                account_id, 
                datetime.now()
            )
            
            # Get activity metrics for last 3 months
            activity_metrics = self.get_account_activity_metrics(
                account_id,
                datetime.now() - timedelta(days=90),
                datetime.now()
            )
            
            # Get 12-month trends
            trends_df = self.calculate_balance_trends(account_id)
            
            return {
                'account_info': account_info,
                'current_balance': current_balance,
                'recent_activity': activity_metrics,
                'trends': {
                    'average_monthly_volume': trends_df['transaction_count'].mean(),
                    'balance_volatility': trends_df['balance_change_pct'].std(),
                    'trend_direction': 'INCREASING' if trends_df['net_change'].mean() > 0 else 'DECREASING',
                    'months_analyzed': len(trends_df)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating account summary for account {account_id}: {str(e)}")
            raise