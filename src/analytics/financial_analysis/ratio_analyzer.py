# src/analytics/financial_analysis/ratio_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader
from .balance_sheet_analyzer import BalanceSheetAnalyzer
from .income_statement_analyzer import IncomeStatementAnalyzer
from .cash_flow_analyzer import CashFlowAnalyzer

class RatioAnalyzer:
    """
    Calculates and analyzes financial ratios using balance sheet, 
    income statement, and cash flow data.
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize with data loader and other analyzers
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.balance_sheet_analyzer = BalanceSheetAnalyzer(data_loader)
        self.income_statement_analyzer = IncomeStatementAnalyzer(data_loader)
        self.cash_flow_analyzer = CashFlowAnalyzer(data_loader)
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.analysis_date = None
        
    def calculate_ratios(self, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Calculate comprehensive financial ratios
        
        Args:
            as_of_date: Analysis date (default: current date)
            
        Returns:
            Dict containing ratio analysis results
        """
        try:
            self.analysis_date = as_of_date or datetime.now()
            
            # Get financial statements data
            bs_data = self.balance_sheet_analyzer.analyze_balance_sheet(self.analysis_date)
            is_data = self.income_statement_analyzer.analyze_income_statement(
                start_date=self.analysis_date - timedelta(days=365),
                end_date=self.analysis_date
            )
            cf_data = self.cash_flow_analyzer.analyze_cash_flow(
                start_date=self.analysis_date - timedelta(days=365),
                end_date=self.analysis_date
            )
            
            # Calculate ratio categories
            self.results = {
                'analysis_date': self.analysis_date,
                'liquidity_ratios': self._calculate_liquidity_ratios(bs_data),
                'profitability_ratios': self._calculate_profitability_ratios(bs_data, is_data),
                'efficiency_ratios': self._calculate_efficiency_ratios(bs_data, is_data),
                'solvency_ratios': self._calculate_solvency_ratios(bs_data),
                'market_ratios': self._calculate_market_ratios(bs_data, is_data),
                'cash_flow_ratios': self._calculate_cash_flow_ratios(bs_data, cf_data),
                'dupont_analysis': self._perform_dupont_analysis(bs_data, is_data),
                'ratio_trends': self._analyze_ratio_trends(),
                'benchmarks': self._get_industry_benchmarks()
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error calculating financial ratios: {str(e)}")
            raise

    def _calculate_liquidity_ratios(self, bs_data: Dict) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        try:
            assets = bs_data['assets']['details']
            liabilities = bs_data['liabilities']['details']
            
            current_assets = assets.get('current_assets', 0)
            current_liabilities = liabilities.get('current_liabilities', 0)
            
            ratios = {}
            
            # Current Ratio
            if current_liabilities != 0:
                ratios['current_ratio'] = current_assets / current_liabilities
            else:
                ratios['current_ratio'] = 0
                
            # Quick Ratio (Acid Test)
            inventory = assets.get('inventory', 0)
            if current_liabilities != 0:
                ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities
            else:
                ratios['quick_ratio'] = 0
                
            # Cash Ratio
            cash = assets.get('cash', 0) + assets.get('cash_equivalents', 0)
            if current_liabilities != 0:
                ratios['cash_ratio'] = cash / current_liabilities
            else:
                ratios['cash_ratio'] = 0
                
            # Working Capital
            ratios['working_capital'] = current_assets - current_liabilities
            
            # Working Capital Ratio
            total_assets = bs_data['assets']['total']
            if total_assets != 0:
                ratios['working_capital_ratio'] = (
                    (current_assets - current_liabilities) / total_assets
                )
            else:
                ratios['working_capital_ratio'] = 0
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity ratios: {str(e)}")
            return {}

    def _calculate_profitability_ratios(self, 
                                      bs_data: Dict,
                                      is_data: Dict) -> Dict[str, float]:
        """Calculate profitability ratios"""
        try:
            # Get required values
            revenue = is_data['revenue']['total']
            gross_profit = is_data['profit_metrics']['gross_profit']
            operating_profit = is_data['profit_metrics']['operating_profit']
            net_profit = is_data['profit_metrics']['net_profit']
            total_assets = bs_data['assets']['total']
            total_equity = bs_data['equity']['total']
            
            ratios = {}
            
            # Margin Ratios
            if revenue != 0:
                ratios['gross_margin'] = (gross_profit / revenue) * 100
                ratios['operating_margin'] = (operating_profit / revenue) * 100
                ratios['net_profit_margin'] = (net_profit / revenue) * 100
                
            # Return Ratios
            if total_assets != 0:
                ratios['return_on_assets'] = (net_profit / total_assets) * 100
                
            if total_equity != 0:
                ratios['return_on_equity'] = (net_profit / total_equity) * 100
                
            # EBITDA Margin
            depreciation = is_data['expenses'].get('depreciation', 0)
            amortization = is_data['expenses'].get('amortization', 0)
            interest = is_data['expenses'].get('interest', 0)
            taxes = is_data['expenses'].get('taxes', 0)
            
            ebitda = operating_profit + depreciation + amortization + interest + taxes
            
            if revenue != 0:
                ratios['ebitda_margin'] = (ebitda / revenue) * 100
                
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating profitability ratios: {str(e)}")
            return {}
        
    
    def _calculate_efficiency_ratios(self,
                                   bs_data: Dict,
                                   is_data: Dict) -> Dict[str, float]:
        """Calculate efficiency (activity) ratios"""
        try:
            # Get required data
            revenue = is_data['revenue']['total']
            total_assets = bs_data['assets']['total']
            current_assets = bs_data.get('assets', {}).get('details', {}).get('current_assets', 0)
            inventory = bs_data.get('assets', {}).get('details', {}).get('inventory', 0)
            
            # Get AR and AP data from GL
            accounts_receivable = self._get_ar_balance()
            accounts_payable = self._get_ap_balance()
            
            # Calculate cost of goods sold from income statement
            cogs = is_data.get('expenses', {}).get('cost_of_sales', 0)
            
            ratios = {}
            
            # Asset Turnover Ratios
            if total_assets != 0:
                ratios['asset_turnover'] = revenue / total_assets
                
            if current_assets != 0:
                ratios['current_asset_turnover'] = revenue / current_assets
                
            # Inventory Turnover
            if inventory != 0:
                ratios['inventory_turnover'] = cogs / inventory
                ratios['days_inventory'] = 365 / ratios['inventory_turnover'] if ratios['inventory_turnover'] != 0 else 0
                
            # Receivables Turnover
            if accounts_receivable != 0:
                ratios['receivables_turnover'] = revenue / accounts_receivable
                ratios['days_receivables'] = 365 / ratios['receivables_turnover'] if ratios['receivables_turnover'] != 0 else 0
                
            # Payables Turnover
            if accounts_payable != 0:
                ratios['payables_turnover'] = cogs / accounts_payable
                ratios['days_payables'] = 365 / ratios['payables_turnover'] if ratios['payables_turnover'] != 0 else 0
                
            # Cash Conversion Cycle
            ratios['cash_conversion_cycle'] = (
                ratios.get('days_inventory', 0) +
                ratios.get('days_receivables', 0) -
                ratios.get('days_payables', 0)
            )
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency ratios: {str(e)}")
            return {}

    def _get_ar_balance(self) -> float:
        """Get accounts receivable balance from GL"""
        try:
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Get AR accounts
            ar_accounts = gl_coa[
                (gl_coa['account_type'] == 'ASSET') &
                (gl_coa['account_category'] == 'RECEIVABLE')
            ]['account_id'].tolist()
            
            # Get AR transactions up to analysis date
            ar_entries = gl_lines[
                (gl_lines['account_id'].isin(ar_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'] <= self.analysis_date) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if ar_entries.empty:
                return 0.0
                
            # Calculate AR balance
            ar_balance = ar_entries['debit_amount'].sum() - ar_entries['credit_amount'].sum()
            return float(ar_balance if ar_balance > 0 else 0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting AR balance: {str(e)}")
            return 0.0

    def _get_ap_balance(self) -> float:
        """Get accounts payable balance from GL"""
        try:
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Get AP accounts
            ap_accounts = gl_coa[
                (gl_coa['account_type'] == 'LIABILITY') &
                (gl_coa['account_category'] == 'PAYABLE')
            ]['account_id'].tolist()
            
            # Get AP transactions up to analysis date
            ap_entries = gl_lines[
                (gl_lines['account_id'].isin(ap_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'] <= self.analysis_date) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if ap_entries.empty:
                return 0.0
                
            # Calculate AP balance
            ap_balance = ap_entries['credit_amount'].sum() - ap_entries['debit_amount'].sum()
            return float(ap_balance if ap_balance > 0 else 0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting AP balance: {str(e)}")
            return 0.0

    def _calculate_solvency_ratios(self, bs_data: Dict) -> Dict[str, float]:
        """Calculate solvency (leverage) ratios"""
        try:
            # Get required values
            total_assets = bs_data['assets']['total']
            total_liabilities = bs_data['liabilities']['total']
            total_equity = bs_data['equity']['total']
            
            long_term_debt = bs_data.get('liabilities', {}).get('details', {}).get('long_term_liabilities', 0)
            
            ratios = {}
            
            # Debt Ratios
            if total_assets != 0:
                ratios['debt_ratio'] = total_liabilities / total_assets
                
            if total_equity != 0:
                ratios['debt_to_equity'] = total_liabilities / total_equity
                ratios['long_term_debt_to_equity'] = long_term_debt / total_equity
                
            # Equity Ratio
            if total_assets != 0:
                ratios['equity_ratio'] = total_equity / total_assets
                
            # Debt to Capital Ratio
            total_capital = long_term_debt + total_equity
            if total_capital != 0:
                ratios['debt_to_capital'] = long_term_debt / total_capital
                
            # Financial Leverage
            if total_equity != 0:
                ratios['financial_leverage'] = total_assets / total_equity
                
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating solvency ratios: {str(e)}")
            return {}
        

    

    def _calculate_market_ratios(self, 
                               bs_data: Dict,
                               is_data: Dict) -> Dict[str, float]:
        """
        Calculate market performance ratios
        Note: Since we're working with Oracle ERP data, we'll calculate 
        book-value based ratios as market data isn't available
        """
        try:
            # Get required values
            net_profit = is_data['profit_metrics']['net_profit']
            equity = bs_data['equity']['total']
            total_shares = self._get_total_shares()
            
            ratios = {}
            
            if total_shares > 0:
                # Book Value Per Share
                ratios['book_value_per_share'] = equity / total_shares
                
                # Earnings Per Share (EPS)
                ratios['earnings_per_share'] = net_profit / total_shares
                
                # Return on Equity (book value based)
                if equity != 0:
                    ratios['return_on_equity'] = (net_profit / equity) * 100
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating market ratios: {str(e)}")
            return {}

    def _calculate_cash_flow_ratios(self,
                                  bs_data: Dict,
                                  cf_data: Dict) -> Dict[str, float]:
        """Calculate cash flow based ratios"""
        try:
            # Get required values
            operating_cash_flow = cf_data['cash_flows']['operating']['net_cash']
            current_liabilities = bs_data['liabilities']['details'].get('current_liabilities', 0)
            total_liabilities = bs_data['liabilities']['total']
            total_assets = bs_data['assets']['total']
            
            ratios = {}
            
            # Operating Cash Flow Ratios
            if current_liabilities != 0:
                ratios['operating_cash_flow_ratio'] = operating_cash_flow / current_liabilities
                
            if total_liabilities != 0:
                ratios['cash_flow_coverage_ratio'] = operating_cash_flow / total_liabilities
                
            if total_assets != 0:
                ratios['cash_flow_to_assets'] = operating_cash_flow / total_assets
                
            # Cash Flow Adequacy
            capital_expenditures = abs(
                cf_data['cash_flows']['investing']['details'].get('asset_purchases', 0)
            )
            if capital_expenditures != 0:
                ratios['cash_flow_adequacy'] = operating_cash_flow / capital_expenditures
            
            # Cash Reinvestment Ratio
            if total_assets != 0:
                ratios['cash_reinvestment_ratio'] = capital_expenditures / total_assets
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating cash flow ratios: {str(e)}")
            return {}

    def _perform_dupont_analysis(self,
                               bs_data: Dict,
                               is_data: Dict) -> Dict[str, float]:
        """
        Perform DuPont analysis breakdown of ROE
        ROE = Net Profit Margin * Asset Turnover * Financial Leverage
        """
        try:
            # Get required values
            net_profit = is_data['profit_metrics']['net_profit']
            revenue = is_data['revenue']['total']
            total_assets = bs_data['assets']['total']
            total_equity = bs_data['equity']['total']
            
            analysis = {}
            
            # Calculate components
            if revenue != 0:
                analysis['net_profit_margin'] = (net_profit / revenue) * 100
            else:
                analysis['net_profit_margin'] = 0
                
            if total_assets != 0:
                analysis['asset_turnover'] = revenue / total_assets
            else:
                analysis['asset_turnover'] = 0
                
            if total_equity != 0:
                analysis['financial_leverage'] = total_assets / total_equity
            else:
                analysis['financial_leverage'] = 0
                
            # Calculate ROE using DuPont formula
            analysis['roe_dupont'] = (
                (analysis['net_profit_margin'] / 100) *
                analysis['asset_turnover'] *
                analysis['financial_leverage'] *
                100  # Convert to percentage
            )
            
            # Add extended DuPont analysis components
            operating_profit = is_data['profit_metrics']['operating_profit']
            if revenue != 0:
                analysis['operating_margin'] = (operating_profit / revenue) * 100
                analysis['tax_burden'] = (net_profit / operating_profit) * 100 if operating_profit != 0 else 0
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error performing DuPont analysis: {str(e)}")
            return {}

    def _get_total_shares(self) -> float:
        """
        Get total number of shares from GL data
        This is a simplified implementation assuming share info is in GL
        """
        try:
            gl_coa = self.data_loader.data['gl_coa']
            gl_lines = self.data_loader.data['gl_journal_lines']
            gl_journals = self.data_loader.data['gl_journals']
            
            # Get share capital accounts
            share_accounts = gl_coa[
                (gl_coa['account_type'] == 'EQUITY') &
                (gl_coa['account_name'].str.contains('SHARE', case=False, na=False))
            ]['account_id'].tolist()
            
            # Get share transactions
            share_entries = gl_lines[
                (gl_lines['account_id'].isin(share_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'] <= self.analysis_date) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if share_entries.empty:
                return 0.0
                
            # Calculate net shares issued (simplified)
            total_shares = share_entries['credit_amount'].sum()
            
            return float(total_shares if total_shares > 0 else 0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting total shares: {str(e)}")
            return 0.0
        
    
    def _analyze_ratio_trends(self, months: int = 12) -> Dict:
        """Analyze trends in key financial ratios"""
        try:
            trend_start = self.analysis_date - timedelta(days=months*30)
            
            trends = {
                'liquidity_trends': [],
                'profitability_trends': [],
                'efficiency_trends': [],
                'solvency_trends': []
            }
            
            # Calculate ratios for each month
            current_date = trend_start
            while current_date <= self.analysis_date:
                self.analysis_date = current_date
                
                # Get financial data for current month
                bs_data = self.balance_sheet_analyzer.analyze_balance_sheet(current_date)
                is_data = self.income_statement_analyzer.analyze_income_statement(
                    start_date=current_date - timedelta(days=30),
                    end_date=current_date
                )
                
                # Calculate key ratios
                liquidity_ratios = self._calculate_liquidity_ratios(bs_data)
                profitability_ratios = self._calculate_profitability_ratios(bs_data, is_data)
                efficiency_ratios = self._calculate_efficiency_ratios(bs_data, is_data)
                solvency_ratios = self._calculate_solvency_ratios(bs_data)
                
                # Store monthly values
                trends['liquidity_trends'].append({
                    'date': current_date,
                    'current_ratio': liquidity_ratios.get('current_ratio', 0),
                    'quick_ratio': liquidity_ratios.get('quick_ratio', 0),
                    'working_capital': liquidity_ratios.get('working_capital', 0)
                })
                
                trends['profitability_trends'].append({
                    'date': current_date,
                    'gross_margin': profitability_ratios.get('gross_margin', 0),
                    'operating_margin': profitability_ratios.get('operating_margin', 0),
                    'net_margin': profitability_ratios.get('net_profit_margin', 0),
                    'roe': profitability_ratios.get('return_on_equity', 0)
                })
                
                trends['efficiency_trends'].append({
                    'date': current_date,
                    'asset_turnover': efficiency_ratios.get('asset_turnover', 0),
                    'inventory_days': efficiency_ratios.get('days_inventory', 0),
                    'receivables_days': efficiency_ratios.get('days_receivables', 0),
                    'payables_days': efficiency_ratios.get('days_payables', 0)
                })
                
                trends['solvency_trends'].append({
                    'date': current_date,
                    'debt_ratio': solvency_ratios.get('debt_ratio', 0),
                    'debt_to_equity': solvency_ratios.get('debt_to_equity', 0),
                    'equity_ratio': solvency_ratios.get('equity_ratio', 0)
                })
                
                # Move to next month
                current_date = current_date + timedelta(days=30)
                
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing ratio trends: {str(e)}")
            return {}

    def _get_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """
        Get industry benchmarks for financial ratios
        Note: In a real implementation, this would fetch from a benchmark database.
        Here we provide typical ranges as an example.
        """
        return {
            'liquidity_benchmarks': {
                'current_ratio': {'low': 1.5, 'medium': 2.0, 'high': 3.0},
                'quick_ratio': {'low': 1.0, 'medium': 1.5, 'high': 2.0},
                'working_capital_ratio': {'low': 0.2, 'medium': 0.3, 'high': 0.4}
            },
            'profitability_benchmarks': {
                'gross_margin': {'low': 20.0, 'medium': 35.0, 'high': 50.0},
                'operating_margin': {'low': 10.0, 'medium': 15.0, 'high': 20.0},
                'net_margin': {'low': 5.0, 'medium': 10.0, 'high': 15.0},
                'return_on_equity': {'low': 10.0, 'medium': 15.0, 'high': 20.0}
            },
            'efficiency_benchmarks': {
                'asset_turnover': {'low': 0.5, 'medium': 1.0, 'high': 2.0},
                'inventory_days': {'low': 30.0, 'medium': 45.0, 'high': 60.0},
                'receivables_days': {'low': 30.0, 'medium': 45.0, 'high': 60.0},
                'payables_days': {'low': 30.0, 'medium': 45.0, 'high': 60.0}
            },
            'solvency_benchmarks': {
                'debt_ratio': {'low': 0.3, 'medium': 0.5, 'high': 0.7},
                'debt_to_equity': {'low': 0.5, 'medium': 1.0, 'high': 1.5},
                'equity_ratio': {'low': 0.3, 'medium': 0.5, 'high': 0.7}
            }
        }

    def get_ratio_summary(self) -> Dict:
        """Get formatted ratio analysis summary"""
        if not self.results:
            raise ValueError("No analysis results available. Run calculate_ratios() first.")
            
        return {
            'analysis_date': self.results['analysis_date'].strftime('%Y-%m-%d'),
            'key_metrics': {
                'liquidity': {
                    'current_ratio': self.results['liquidity_ratios'].get('current_ratio', 0),
                    'quick_ratio': self.results['liquidity_ratios'].get('quick_ratio', 0),
                    'working_capital': self.results['liquidity_ratios'].get('working_capital', 0)
                },
                'profitability': {
                    'gross_margin': self.results['profitability_ratios'].get('gross_margin', 0),
                    'operating_margin': self.results['profitability_ratios'].get('operating_margin', 0),
                    'net_margin': self.results['profitability_ratios'].get('net_profit_margin', 0),
                    'roe': self.results['profitability_ratios'].get('return_on_equity', 0)
                },
                'efficiency': {
                    'asset_turnover': self.results['efficiency_ratios'].get('asset_turnover', 0),
                    'cash_conversion_cycle': self.results['efficiency_ratios'].get('cash_conversion_cycle', 0)
                },
                'solvency': {
                    'debt_ratio': self.results['solvency_ratios'].get('debt_ratio', 0),
                    'debt_to_equity': self.results['solvency_ratios'].get('debt_to_equity', 0)
                }
            },
            'dupont_analysis': self.results['dupont_analysis'],
            'ratio_trends': self.results['ratio_trends'],
            'performance_vs_benchmarks': self._compare_with_benchmarks()
        }
        
    def _compare_with_benchmarks(self) -> Dict:
        """Compare current ratios with industry benchmarks"""
        try:
            benchmarks = self._get_industry_benchmarks()
            comparison = {}
            
            for category, metrics in self.results.items():
                if not isinstance(metrics, dict) or category not in [
                    'liquidity_ratios', 'profitability_ratios', 
                    'efficiency_ratios', 'solvency_ratios'
                ]:
                    continue
                    
                category_name = category.replace('_ratios', '')
                benchmark_category = f"{category_name}_benchmarks"
                
                if benchmark_category not in benchmarks:
                    continue
                    
                comparison[category_name] = {}
                
                for ratio, value in metrics.items():
                    if ratio in benchmarks[benchmark_category]:
                        benchmark = benchmarks[benchmark_category][ratio]
                        rating = self._get_benchmark_rating(value, benchmark)
                        
                        comparison[category_name][ratio] = {
                            'value': value,
                            'benchmark': benchmark,
                            'rating': rating
                        }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing with benchmarks: {str(e)}")
            return {}
            
    def _get_benchmark_rating(self, value: float, benchmark: Dict[str, float]) -> str:
        """Get rating based on benchmark comparison"""
        if value < benchmark['low']:
            return 'BELOW_AVERAGE'
        elif value < benchmark['medium']:
            return 'AVERAGE'
        elif value < benchmark['high']:
            return 'ABOVE_AVERAGE'
        else:
            return 'EXCELLENT'