# src/analytics/financial_analysis/balance_sheet_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class BalanceSheetAnalyzer:
    """
    Analyzes balance sheet accounts and generates financial analysis
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
        self.analysis_date = None
        self.results = {}

    def analyze_balance_sheet(self, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Perform comprehensive balance sheet analysis
        
        Args:
            as_of_date: Analysis date, defaults to current date
            
        Returns:
            Dict containing balance sheet analysis results
        """
        try:
            self.analysis_date = as_of_date or datetime.now()
            
            # Calculate main components
            assets = self._calculate_assets()
            liabilities = self._calculate_liabilities()
            equity = self._calculate_equity()
            
            # Calculate totals and key metrics
            total_assets = sum(assets.values())
            total_liabilities = sum(liabilities.values())
            total_equity = sum(equity.values())
            
            # Validate balance sheet equation
            if not np.isclose(total_assets, total_liabilities + total_equity, rtol=0.01):
                self.logger.warning("Balance sheet equation doesn't balance exactly")
                
            self.results = {
                'analysis_date': self.analysis_date,
                'assets': {
                    'details': assets,
                    'total': total_assets
                },
                'liabilities': {
                    'details': liabilities,
                    'total': total_liabilities
                },
                'equity': {
                    'details': equity,
                    'total': total_equity
                },
                'metrics': self._calculate_balance_sheet_metrics(
                    assets, liabilities, equity
                ),
                'trend_analysis': self._analyze_trends()
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in balance sheet analysis: {str(e)}")
            raise

    def _calculate_assets(self) -> Dict[str, float]:
        """Calculate detailed asset balances"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Filter for asset accounts
            asset_accounts = gl_coa[gl_coa['account_type'] == 'ASSET']
            
            # Initialize results
            assets = {
                'current_assets': 0.0,
                'fixed_assets': 0.0,
                'other_assets': 0.0
            }
            
            # Calculate balances for each asset account
            for _, account in asset_accounts.iterrows():
                balance = self._get_account_balance(
                    account['account_id'],
                    gl_lines,
                    gl_journals
                )
                
                # Categorize based on account category
                if account['account_category'] == 'CURRENT':
                    assets['current_assets'] += balance
                elif account['account_category'] == 'FIXED':
                    assets['fixed_assets'] += balance
                else:
                    assets['other_assets'] += balance
                    
            return assets
            
        except Exception as e:
            self.logger.error(f"Error calculating assets: {str(e)}")
            raise

    def _calculate_liabilities(self) -> Dict[str, float]:
        """Calculate detailed liability balances"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Filter for liability accounts
            liability_accounts = gl_coa[gl_coa['account_type'] == 'LIABILITY']
            
            # Initialize results
            liabilities = {
                'current_liabilities': 0.0,
                'long_term_liabilities': 0.0,
                'other_liabilities': 0.0
            }
            
            # Calculate balances for each liability account
            for _, account in liability_accounts.iterrows():
                balance = self._get_account_balance(
                    account['account_id'],
                    gl_lines,
                    gl_journals
                )
                
                # Categorize based on account category
                if account['account_category'] == 'CURRENT':
                    liabilities['current_liabilities'] += balance
                elif account['account_category'] == 'LONG_TERM':
                    liabilities['long_term_liabilities'] += balance
                else:
                    liabilities['other_liabilities'] += balance
                    
            return liabilities
            
        except Exception as e:
            self.logger.error(f"Error calculating liabilities: {str(e)}")
            raise

    def _calculate_equity(self) -> Dict[str, float]:
        """Calculate detailed equity balances"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Filter for equity accounts
            equity_accounts = gl_coa[gl_coa['account_type'] == 'EQUITY']
            
            # Initialize results
            equity = {
                'capital': 0.0,
                'retained_earnings': 0.0,
                'other_equity': 0.0
            }
            
            # Calculate balances for each equity account
            for _, account in equity_accounts.iterrows():
                balance = self._get_account_balance(
                    account['account_id'],
                    gl_lines,
                    gl_journals
                )
                
                # Categorize based on account category
                if account['account_category'] == 'CAPITAL':
                    equity['capital'] += balance
                elif account['account_category'] == 'RETAINED_EARNINGS':
                    equity['retained_earnings'] += balance
                else:
                    equity['other_equity'] += balance
                    
            return equity
            
        except Exception as e:
            self.logger.error(f"Error calculating equity: {str(e)}")
            raise

    def _get_account_balance(self, 
                           account_id: int,
                           gl_lines: pd.DataFrame,
                           gl_journals: pd.DataFrame) -> float:
        """
        Calculate balance for a specific account
        
        Args:
            account_id: GL account ID
            gl_lines: Journal lines DataFrame
            gl_journals: Journal headers DataFrame
            
        Returns:
            Account balance as float
        """
        try:
            # Filter for specific account and posted journals
            account_lines = gl_lines[gl_lines['account_id'] == account_id]
            posted_journals = gl_journals[
                (gl_journals['status'] == 'POSTED') &
                (gl_journals['journal_date'] <= self.analysis_date)
            ]
            
            # Join with posted journals
            valid_lines = account_lines[
                account_lines['journal_id'].isin(posted_journals['journal_id'])
            ]
            
            if valid_lines.empty:
                return 0.0
                
            # Calculate net balance
            debits = valid_lines['debit_amount'].sum()
            credits = valid_lines['credit_amount'].sum()
            
            return debits - credits
            
        except Exception as e:
            self.logger.error(f"Error calculating account balance: {str(e)}")
            raise

    def _calculate_balance_sheet_metrics(self,
                                       assets: Dict[str, float],
                                       liabilities: Dict[str, float],
                                       equity: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate key balance sheet metrics and ratios
        
        Args:
            assets: Asset balances by category
            liabilities: Liability balances by category
            equity: Equity balances by category
            
        Returns:
            Dict of balance sheet metrics
        """
        try:
            metrics = {}
            
            # Working capital
            metrics['working_capital'] = (
                assets['current_assets'] - liabilities['current_liabilities']
            )
            
            # Current ratio
            metrics['current_ratio'] = (
                assets['current_assets'] / liabilities['current_liabilities']
                if liabilities['current_liabilities'] != 0 else float('inf')
            )
            
            # Debt to equity ratio
            total_liabilities = sum(liabilities.values())
            total_equity = sum(equity.values())
            metrics['debt_to_equity'] = (
                total_liabilities / total_equity
                if total_equity != 0 else float('inf')
            )
            
            # Asset utilization
            total_assets = sum(assets.values())
            metrics['fixed_asset_ratio'] = (
                assets['fixed_assets'] / total_assets
                if total_assets != 0 else 0
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating balance sheet metrics: {str(e)}")
            raise

    def _analyze_trends(self, months: int = 12) -> Dict:
        """
        Analyze balance sheet trends over time
        
        Args:
            months: Number of months to analyze
            
        Returns:
            Dict containing trend analysis
        """
        try:
            start_date = self.analysis_date - timedelta(days=months*30)
            
            # Initialize trend data structure
            trends = {
                'total_assets': [],
                'total_liabilities': [],
                'total_equity': [],
                'working_capital': [],
                'current_ratio': []
            }
            
            # Calculate metrics for each month
            current_date = start_date
            while current_date <= self.analysis_date:
                self.analysis_date = current_date
                
                # Calculate components
                assets = self._calculate_assets()
                liabilities = self._calculate_liabilities()
                equity = self._calculate_equity()
                
                # Store monthly values
                trends['total_assets'].append({
                    'date': current_date,
                    'value': sum(assets.values())
                })
                
                trends['total_liabilities'].append({
                    'date': current_date,
                    'value': sum(liabilities.values())
                })
                
                trends['total_equity'].append({
                    'date': current_date,
                    'value': sum(equity.values())
                })
                
                working_capital = (
                    assets['current_assets'] - liabilities['current_liabilities']
                )
                trends['working_capital'].append({
                    'date': current_date,
                    'value': working_capital
                })
                
                current_ratio = (
                    assets['current_assets'] / liabilities['current_liabilities']
                    if liabilities['current_liabilities'] != 0 else float('inf')
                )
                trends['current_ratio'].append({
                    'date': current_date,
                    'value': current_ratio
                })
                
                # Move to next month
                current_date = current_date + timedelta(days=30)
                
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {str(e)}")
            raise

    def get_balance_sheet_summary(self) -> Dict:
        """Get formatted balance sheet summary"""
        if not self.results:
            raise ValueError("No analysis results available. Run analyze_balance_sheet() first.")
            
        return {
            'date': self.analysis_date.strftime('%Y-%m-%d'),
            'assets': {
                'total': self.results['assets']['total'],
                'breakdown': self.results['assets']['details']
            },
            'liabilities': {
                'total': self.results['liabilities']['total'],
                'breakdown': self.results['liabilities']['details']
            },
            'equity': {
                'total': self.results['equity']['total'],
                'breakdown': self.results['equity']['details']
            },
            'key_metrics': self.results['metrics']
        }