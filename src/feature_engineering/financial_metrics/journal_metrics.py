# src/feature_engineering/financial_metrics/journal_metrics.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

from src.utils.database_utils import DatabaseConnector

class JournalMetricsCalculator:
    """
    Class to calculate and analyze journal entry metrics including:
    - Journal balancing analysis
    - Entry patterns and frequency
    - Journal line analytics
    - Period-based metrics
    """
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger for journal metrics"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Only add handlers if there aren't any
        if not logger.handlers:
            # Create file handler using existing logs directory
            log_file = os.path.join('logs', 'feature_engineering', 'journal_metrics.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)

            # Add cleanup for handler closure
            def cleanup():
                for handler in logger.handlers:
                    handler.close()
            import atexit
            atexit.register(cleanup)
            
        return logger

    def calculate_journal_metrics(self, journal_id: int) -> Dict[str, any]:
        """
        Calculate comprehensive metrics for a specific journal entry
        
        Args:
            journal_id: The journal entry ID to analyze
            
        Returns:
            Dict containing journal metrics
        """
        try:
            # Get journal header information
            journal_query = """
                SELECT 
                    j.journal_id,
                    j.journal_date,
                    j.journal_number,
                    j.status,
                    j.period_name,
                    COUNT(DISTINCT l.account_id) as unique_accounts,
                    COUNT(l.line_id) as line_count,
                    SUM(l.debit_amount) as total_debits,
                    SUM(l.credit_amount) as total_credits,
                    MAX(l.debit_amount) as max_debit,
                    MAX(l.credit_amount) as max_credit
                FROM GL_JOURNALS j
                LEFT JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                WHERE j.journal_id = :journal_id
                GROUP BY 
                    j.journal_id,
                    j.journal_date,
                    j.journal_number,
                    j.status,
                    j.period_name
            """
            
            df = self.db.execute_query(
                journal_query,
                params={'journal_id': journal_id}
            )
            
            if df.empty:
                raise ValueError(f"Journal {journal_id} not found")
                
            journal_info = df.iloc[0].to_dict()
            
            # Calculate balance and related metrics
            balance = journal_info['total_debits'] - journal_info['total_credits']
            is_balanced = abs(balance) < 0.01  # Small tolerance for rounding
            
            return {
                'journal_info': {
                    'journal_id': journal_info['journal_id'],
                    'journal_number': journal_info['journal_number'],
                    'journal_date': journal_info['journal_date'],
                    'period_name': journal_info['period_name'],
                    'status': journal_info['status']
                },
                'balance_metrics': {
                    'total_debits': float(journal_info['total_debits']),
                    'total_credits': float(journal_info['total_credits']),
                    'balance': float(balance),
                    'is_balanced': is_balanced
                },
                'composition_metrics': {
                    'line_count': journal_info['line_count'],
                    'unique_accounts': journal_info['unique_accounts'],
                    'max_debit': float(journal_info['max_debit']),
                    'max_credit': float(journal_info['max_credit']),
                    'complexity_score': self._calculate_complexity_score(
                        journal_info['line_count'],
                        journal_info['unique_accounts']
                    )
                },
                'risk_indicators': self._calculate_risk_indicators(journal_info)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating journal metrics for journal {journal_id}: {str(e)}")
            raise

    def _calculate_complexity_score(self, line_count: int, unique_accounts: int) -> str:
        """Calculate journal entry complexity score"""
        complexity_score = (line_count * 0.6) + (unique_accounts * 0.4)
        
        if complexity_score <= 2:
            return 'SIMPLE'
        elif complexity_score <= 5:
            return 'MODERATE'
        elif complexity_score <= 10:
            return 'COMPLEX'
        else:
            return 'VERY_COMPLEX'

    def _calculate_risk_indicators(self, journal_info: Dict) -> Dict[str, any]:
        """Calculate risk indicators for journal entry"""
        risk_indicators = {
            'balance_risk': 'HIGH' if not abs(journal_info['total_debits'] - journal_info['total_credits']) < 0.01 else 'LOW',
            'size_risk': self._assess_size_risk(journal_info['total_debits']),
            'complexity_risk': self._assess_complexity_risk(journal_info['line_count'], journal_info['unique_accounts'])
        }
        
        # Calculate overall risk
        risk_scores = {
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3
        }
        
        avg_risk_score = sum(risk_scores[risk] for risk in risk_indicators.values()) / len(risk_indicators)
        
        if avg_risk_score <= 1.5:
            risk_indicators['overall_risk'] = 'LOW'
        elif avg_risk_score <= 2.5:
            risk_indicators['overall_risk'] = 'MEDIUM'
        else:
            risk_indicators['overall_risk'] = 'HIGH'
            
        return risk_indicators

    def _assess_size_risk(self, total_amount: float) -> str:
        """Assess risk based on journal entry size"""
        if total_amount <= 1000:
            return 'LOW'
        elif total_amount <= 10000:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _assess_complexity_risk(self, line_count: int, unique_accounts: int) -> str:
        """Assess risk based on journal complexity"""
        complexity_score = (line_count * 0.6) + (unique_accounts * 0.4)
        
        if complexity_score <= 3:
            return 'LOW'
        elif complexity_score <= 7:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def analyze_journal_patterns(self, period: str) -> Dict[str, any]:
        """
        Analyze journal entry patterns for a specific period
        
        Args:
            period: The period to analyze (format: YYYY-MM)
            
        Returns:
            Dict containing journal pattern analysis
        """
        try:
            pattern_query = """
                WITH JournalStats AS (
                    SELECT 
                        j.journal_id,
                        COUNT(l.line_id) as line_count,
                        COUNT(DISTINCT l.account_id) as account_count,
                        SUM(l.debit_amount) as total_amount,
                        CASE 
                            WHEN ABS(SUM(l.debit_amount) - SUM(l.credit_amount)) < 0.01 
                            THEN 1 ELSE 0 
                        END as is_balanced
                    FROM GL_JOURNALS j
                    JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                    WHERE j.period_name = :period
                    GROUP BY j.journal_id
                )
                SELECT
                    COUNT(*) as total_journals,
                    AVG(line_count) as avg_lines_per_journal,
                    AVG(account_count) as avg_accounts_per_journal,
                    SUM(CASE WHEN is_balanced = 1 THEN 1 ELSE 0 END) as balanced_journals,
                    AVG(total_amount) as avg_journal_amount,
                    MAX(total_amount) as max_journal_amount
                FROM JournalStats
            """
            
            df = self.db.execute_query(
                pattern_query,
                params={'period': period}
            )
            
            if df.empty:
                return {
                    'period': period,
                    'status': 'NO_DATA'
                }
                
            stats = df.iloc[0].to_dict()
            
            return {
                'period': period,
                'status': 'ANALYZED',
                'volume_metrics': {
                    'total_journals': int(stats['total_journals']),
                    'balanced_journals': int(stats['balanced_journals']),
                    'balance_rate': (stats['balanced_journals'] / stats['total_journals'] * 100 
                                   if stats['total_journals'] > 0 else 0)
                },
                'composition_metrics': {
                    'avg_lines_per_journal': round(float(stats['avg_lines_per_journal']), 2),
                    'avg_accounts_per_journal': round(float(stats['avg_accounts_per_journal']), 2)
                },
                'value_metrics': {
                    'avg_journal_amount': float(stats['avg_journal_amount']),
                    'max_journal_amount': float(stats['max_journal_amount'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing journal patterns for period {period}: {str(e)}")
            raise

    def get_unbalanced_journals(self, start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              tolerance: float = 0.01) -> List[Dict[str, any]]:
        """
        Get list of unbalanced journal entries
        
        Args:
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            tolerance: Balance difference tolerance
            
        Returns:
            List of unbalanced journal entries with details
        """
        try:
            query = """
                WITH JournalBalances AS (
                    SELECT 
                        j.journal_id,
                        j.journal_number,
                        j.journal_date,
                        j.status,
                        SUM(l.debit_amount) as total_debits,
                        SUM(l.credit_amount) as total_credits,
                        ABS(SUM(l.debit_amount) - SUM(l.credit_amount)) as balance_difference
                    FROM GL_JOURNALS j
                    JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                    WHERE 1=1
                        {date_filter}
                    GROUP BY 
                        j.journal_id,
                        j.journal_number,
                        j.journal_date,
                        j.status
                    HAVING ABS(SUM(l.debit_amount) - SUM(l.credit_amount)) > :tolerance
                )
                SELECT *
                FROM JournalBalances
                ORDER BY balance_difference DESC
            """
            
            date_filter = ""
            params = {'tolerance': tolerance}
            
            if start_date and end_date:
                date_filter = "AND j.journal_date BETWEEN :start_date AND :end_date"
                params.update({
                    'start_date': start_date,
                    'end_date': end_date
                })
                
            query = query.format(date_filter=date_filter)
            
            df = self.db.execute_query(query, params=params)
            
            if df.empty:
                return []
                
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error getting unbalanced journals: {str(e)}")
            raise