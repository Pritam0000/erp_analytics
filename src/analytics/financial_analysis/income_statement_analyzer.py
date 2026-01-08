# src/analytics/financial_analysis/income_statement_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class IncomeStatementAnalyzer:
    """
    Analyzes income statement accounts and generates profitability analysis
    Integrates with existing GL data structure from data_loader.py
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize with data loader instance
        
        Args:
            data_loader: Instance of FinancialDataLoader with required GL data
        """
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        self.start_date = None
        self.end_date = None
        self.results = {}

    def analyze_income_statement(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict:
        """
        Perform comprehensive income statement analysis
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dict containing income statement analysis
        """
        try:
            self.end_date = end_date or datetime.now()
            self.start_date = start_date or (
                self.end_date.replace(day=1) - timedelta(days=1)
            ).replace(day=1)  # Default to start of previous month
            
            # Calculate main components
            revenue = self._calculate_revenue()
            expenses = self._calculate_expenses()
            
            # Calculate key profit metrics
            gross_profit = revenue['total'] - expenses['cost_of_sales']
            operating_profit = gross_profit - expenses['operating']
            net_profit = operating_profit - expenses['other']
            
            self.results = {
                'period': {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                },
                'revenue': revenue,
                'expenses': expenses,
                'profit_metrics': {
                    'gross_profit': gross_profit,
                    'operating_profit': operating_profit,
                    'net_profit': net_profit
                },
                'performance_metrics': self._calculate_performance_metrics(
                    revenue, expenses, gross_profit, operating_profit, net_profit
                ),
                'trend_analysis': self._analyze_trends()
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in income statement analysis: {str(e)}")
            raise

    def _calculate_revenue(self) -> Dict[str, float]:
        """Calculate detailed revenue breakdown"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Filter for revenue accounts
            revenue_accounts = gl_coa[gl_coa['account_type'] == 'REVENUE']
            
            # Initialize revenue categories
            revenue = {
                'operating_revenue': 0.0,
                'other_revenue': 0.0,
                'total': 0.0
            }
            
            # Calculate revenue by category
            for _, account in revenue_accounts.iterrows():
                balance = self._get_account_activity(
                    account['account_id'],
                    gl_lines,
                    gl_journals
                )
                
                if account['account_category'] == 'OPERATING':
                    revenue['operating_revenue'] += balance
                else:
                    revenue['other_revenue'] += balance
                    
            revenue['total'] = revenue['operating_revenue'] + revenue['other_revenue']
            return revenue
            
        except Exception as e:
            self.logger.error(f"Error calculating revenue: {str(e)}")
            raise

    def _calculate_expenses(self) -> Dict[str, float]:
        """Calculate detailed expense breakdown"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Filter for expense accounts
            expense_accounts = gl_coa[gl_coa['account_type'] == 'EXPENSE']
            
            # Initialize expense categories
            expenses = {
                'cost_of_sales': 0.0,
                'operating': 0.0,
                'other': 0.0,
                'total': 0.0
            }
            
            # Calculate expenses by category
            for _, account in expense_accounts.iterrows():
                balance = self._get_account_activity(
                    account['account_id'],
                    gl_lines,
                    gl_journals
                )
                
                if account['account_category'] == 'COST_OF_SALES':
                    expenses['cost_of_sales'] += balance
                elif account['account_category'] == 'OPERATING':
                    expenses['operating'] += balance
                else:
                    expenses['other'] += balance
                    
            expenses['total'] = sum(
                value for key, value in expenses.items() 
                if key != 'total'
            )
            return expenses
            
        except Exception as e:
            self.logger.error(f"Error calculating expenses: {str(e)}")
            raise

    def _get_account_activity(self, 
                            account_id: int,
                            gl_lines: pd.DataFrame,
                            gl_journals: pd.DataFrame) -> float:
        """
        Calculate account activity for a specific period
        
        Args:
            account_id: GL account ID
            gl_lines: Journal lines DataFrame
            gl_journals: Journal headers DataFrame
            
        Returns:
            Account activity as float
        """
        try:
            # Filter for specific account and posted journals in period
            account_lines = gl_lines[gl_lines['account_id'] == account_id]
            posted_journals = gl_journals[
                (gl_journals['status'] == 'POSTED') &
                (gl_journals['journal_date'].between(self.start_date, self.end_date))
            ]
            
            # Join with posted journals
            valid_lines = account_lines[
                account_lines['journal_id'].isin(posted_journals['journal_id'])
            ]
            
            if valid_lines.empty:
                return 0.0
                
            # Calculate net activity
            debits = valid_lines['debit_amount'].sum()
            credits = valid_lines['credit_amount'].sum()
            
            return abs(credits - debits)  # Take absolute value for both revenue and expense
            
        except Exception as e:
            self.logger.error(f"Error calculating account activity: {str(e)}")
            raise

    def _calculate_performance_metrics(self,
                                    revenue: Dict[str, float],
                                    expenses: Dict[str, float],
                                    gross_profit: float,
                                    operating_profit: float,
                                    net_profit: float) -> Dict[str, float]:
        """
        Calculate key performance metrics and ratios
        
        Args:
            revenue: Revenue breakdown
            expenses: Expense breakdown
            gross_profit: Gross profit amount
            operating_profit: Operating profit amount
            net_profit: Net profit amount
            
        Returns:
            Dict of performance metrics
        """
        try:
            metrics = {}
            
            if revenue['total'] != 0:
                # Margin ratios
                metrics['gross_margin'] = (gross_profit / revenue['total']) * 100
                metrics['operating_margin'] = (operating_profit / revenue['total']) * 100
                metrics['net_margin'] = (net_profit / revenue['total']) * 100
                
                # Expense ratios
                metrics['cost_of_sales_ratio'] = (expenses['cost_of_sales'] / revenue['total']) * 100
                metrics['operating_expense_ratio'] = (expenses['operating'] / revenue['total']) * 100
                metrics['other_expense_ratio'] = (expenses['other'] / revenue['total']) * 100
            else:
                # Set to 0 if no revenue
                metrics.update({
                    'gross_margin': 0,
                    'operating_margin': 0,
                    'net_margin': 0,
                    'cost_of_sales_ratio': 0,
                    'operating_expense_ratio': 0,
                    'other_expense_ratio': 0
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    def _analyze_trends(self, months: int = 12) -> Dict:
        """
        Analyze income statement trends over time
        
        Args:
            months: Number of months to analyze
            
        Returns:
            Dict containing trend analysis
        """
        try:
            # Calculate start date for trend analysis
            trend_start = self.end_date - timedelta(days=months*30)
            
            trends = {
                'revenue': [],
                'gross_profit': [],
                'operating_profit': [],
                'net_profit': [],
                'margins': []
            }
            
            # Calculate metrics for each month
            current_date = trend_start
            while current_date <= self.end_date:
                # Set analysis period to current month
                self.start_date = current_date.replace(day=1)
                self.end_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                
                # Calculate monthly metrics
                revenue = self._calculate_revenue()
                expenses = self._calculate_expenses()
                
                gross_profit = revenue['total'] - expenses['cost_of_sales']
                operating_profit = gross_profit - expenses['operating']
                net_profit = operating_profit - expenses['other']
                
                # Store monthly values
                trends['revenue'].append({
                    'date': current_date,
                    'value': revenue['total']
                })
                
                trends['gross_profit'].append({
                    'date': current_date,
                    'value': gross_profit
                })
                
                trends['operating_profit'].append({
                    'date': current_date,
                    'value': operating_profit
                })
                
                trends['net_profit'].append({
                    'date': current_date,
                    'value': net_profit
                })
                
                # Calculate and store margins
                if revenue['total'] != 0:
                    trends['margins'].append({
                        'date': current_date,
                        'gross_margin': (gross_profit / revenue['total']) * 100,
                        'operating_margin': (operating_profit / revenue['total']) * 100,
                        'net_margin': (net_profit / revenue['total']) * 100
                    })
                
                # Move to next month
                current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {str(e)}")
            raise

    def get_income_statement_summary(self) -> Dict:
        """Get formatted income statement summary"""
        if not self.results:
            raise ValueError("No analysis results available. Run analyze_income_statement() first.")
            
        return {
            'period': {
                'start_date': self.results['period']['start_date'].strftime('%Y-%m-%d'),
                'end_date': self.results['period']['end_date'].strftime('%Y-%m-%d')
            },
            'revenue': self.results['revenue'],
            'expenses': self.results['expenses'],
            'profit': {
                'gross_profit': self.results['profit_metrics']['gross_profit'],
                'operating_profit': self.results['profit_metrics']['operating_profit'],
                'net_profit': self.results['profit_metrics']['net_profit']
            },
            'key_metrics': self.results['performance_metrics']
        }