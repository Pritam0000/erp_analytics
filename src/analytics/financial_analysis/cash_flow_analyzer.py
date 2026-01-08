# src/analytics/financial_analysis/cash_flow_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from src.analytics.utils.model_utils import ModelUtils
from src.data_loading.data_loader import FinancialDataLoader

class CashFlowAnalyzer:
    """
    Analyzes cash flow using GL transactions and AP/AR data
    Integrates with existing GL, AP, and AR data structure
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize with data loader instance
        
        Args:
            data_loader: Instance of FinancialDataLoader with required data
        """
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        self.start_date = None
        self.end_date = None
        self.results = {}
        
        # Define cash and cash equivalent account types
        self.cash_account_types = ['CASH', 'BANK', 'CASH_EQUIVALENT']
        
    def analyze_cash_flow(self,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict:
        """
        Perform comprehensive cash flow analysis
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dict containing cash flow analysis
        """
        try:
            self.end_date = end_date or datetime.now()
            self.start_date = start_date or (
                self.end_date.replace(day=1) - timedelta(days=1)
            ).replace(day=1)
            
            # Calculate main cash flow components
            operating_cash_flow = self._calculate_operating_cash_flow()
            investing_cash_flow = self._calculate_investing_cash_flow()
            financing_cash_flow = self._calculate_financing_cash_flow()
            
            # Calculate net cash flow
            net_cash_flow = (
                operating_cash_flow['net_cash'] +
                investing_cash_flow['net_cash'] +
                financing_cash_flow['net_cash']
            )
            
            # Get starting and ending cash positions
            starting_cash = self._get_cash_position(self.start_date)
            ending_cash = self._get_cash_position(self.end_date)
            
            self.results = {
                'period': {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                },
                'cash_positions': {
                    'starting_cash': starting_cash,
                    'ending_cash': ending_cash,
                    'net_change': ending_cash - starting_cash
                },
                'cash_flows': {
                    'operating': operating_cash_flow,
                    'investing': investing_cash_flow,
                    'financing': financing_cash_flow,
                    'net_cash_flow': net_cash_flow
                },
                'metrics': self._calculate_cash_flow_metrics(
                    operating_cash_flow,
                    investing_cash_flow,
                    financing_cash_flow,
                    starting_cash
                ),
                'trend_analysis': self._analyze_trends()
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in cash flow analysis: {str(e)}")
            raise

    

    def _calculate_operating_cash_flow(self) -> Dict[str, float]:
        """Calculate operating cash flow components"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            ap_payments = self.data_loader.data['ap_payments']
            
            operating_flows = {
                'collections_from_customers': self._get_customer_collections(),
                'payments_to_suppliers': self._get_supplier_payments(ap_payments),
                'operating_expenses': self._get_operating_expenses(
                    gl_coa, gl_journals, gl_lines
                ),
                'taxes_paid': self._get_tax_payments(
                    gl_coa, gl_journals, gl_lines
                ),
                'other_operating': self._get_other_operating_flows(
                    gl_coa, gl_journals, gl_lines
                )
            }
            
            # Calculate net operating cash flow
            operating_flows['net_cash'] = (
                operating_flows['collections_from_customers']
                - abs(operating_flows['payments_to_suppliers'])
                - abs(operating_flows['operating_expenses'])
                - abs(operating_flows['taxes_paid'])
                + operating_flows['other_operating']
            )
            
            return operating_flows
            
        except Exception as e:
            self.logger.error(f"Error calculating operating cash flow: {str(e)}")
            raise
            
    def _get_customer_collections(self) -> float:
        """Calculate cash collections from customers"""
        try:
            # Using AP_PAYMENTS table as a proxy for collections
            # In a real system, you'd use AR_RECEIPTS or similar
            payments = self.data_loader.data['ap_payments']
            
            collections = payments[
                (payments['payment_date'].between(self.start_date, self.end_date)) &
                (payments['status'] == 'POSTED')
            ]['amount'].sum()
            
            return float(collections or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating customer collections: {str(e)}")
            return 0.0

    def _get_supplier_payments(self, ap_payments: pd.DataFrame) -> float:
        """Calculate payments made to suppliers"""
        try:
            supplier_payments = ap_payments[
                (ap_payments['payment_date'].between(self.start_date, self.end_date)) &
                (ap_payments['status'] == 'POSTED')
            ]['amount'].sum()
            
            return float(supplier_payments or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating supplier payments: {str(e)}")
            return 0.0

    def _get_operating_expenses(self,
                              gl_coa: pd.DataFrame,
                              gl_journals: pd.DataFrame,
                              gl_lines: pd.DataFrame) -> float:
        """Calculate cash operating expenses"""
        try:
            # Get operating expense accounts
            operating_accounts = gl_coa[
                (gl_coa['account_type'] == 'EXPENSE') &
                (gl_coa['account_category'] == 'OPERATING')
            ]['account_id'].tolist()
            
            # Get related journal entries
            operating_entries = gl_lines[
                (gl_lines['account_id'].isin(operating_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if operating_entries.empty:
                return 0.0
                
            # Calculate net cash impact
            expenses = operating_entries['debit_amount'].sum() - operating_entries['credit_amount'].sum()
            
            return float(expenses if expenses > 0 else 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating operating expenses: {str(e)}")
            return 0.0

    def _get_tax_payments(self,
                         gl_coa: pd.DataFrame,
                         gl_journals: pd.DataFrame,
                         gl_lines: pd.DataFrame) -> float:
        """Calculate tax payments"""
        try:
            # Get tax-related accounts
            tax_accounts = gl_coa[
                gl_coa['account_name'].str.contains('TAX', case=False, na=False)
            ]['account_id'].tolist()
            
            # Get tax-related journal entries
            tax_entries = gl_lines[
                (gl_lines['account_id'].isin(tax_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if tax_entries.empty:
                return 0.0
                
            # Calculate net tax payments
            tax_payments = tax_entries['debit_amount'].sum() - tax_entries['credit_amount'].sum()
            
            return float(tax_payments if tax_payments > 0 else 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating tax payments: {str(e)}")
            return 0.0

    def _get_other_operating_flows(self,
                                 gl_coa: pd.DataFrame,
                                 gl_journals: pd.DataFrame,
                                 gl_lines: pd.DataFrame) -> float:
        """Calculate other operating cash flows"""
        try:
            # Get other operating accounts (exclude main categories already counted)
            excluded_categories = ['OPERATING', 'TAX', 'INVESTING', 'FINANCING']
            other_accounts = gl_coa[
                ~gl_coa['account_category'].isin(excluded_categories)
            ]['account_id'].tolist()
            
            # Get related journal entries
            other_entries = gl_lines[
                (gl_lines['account_id'].isin(other_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if other_entries.empty:
                return 0.0
                
            # Calculate net other flows
            other_flows = other_entries['credit_amount'].sum() - other_entries['debit_amount'].sum()
            
            return float(other_flows)
            
        except Exception as e:
            self.logger.error(f"Error calculating other operating flows: {str(e)}")
            return 0.0
        

            
    def _calculate_investing_cash_flow(self) -> Dict[str, float]:
        """Calculate investing cash flow components"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            fa_assets = self.data_loader.data['fa_assets']
            
            investing_flows = {
                'asset_purchases': self._get_asset_purchases(
                    gl_coa, gl_journals, gl_lines, fa_assets
                ),
                'asset_sales': self._get_asset_sales(
                    gl_coa, gl_journals, gl_lines, fa_assets
                ),
                'investment_purchases': self._get_investment_purchases(
                    gl_coa, gl_journals, gl_lines
                ),
                'investment_sales': self._get_investment_sales(
                    gl_coa, gl_journals, gl_lines
                ),
                'other_investing': self._get_other_investing_flows(
                    gl_coa, gl_journals, gl_lines
                )
            }
            
            # Calculate net investing cash flow
            investing_flows['net_cash'] = (
                -abs(investing_flows['asset_purchases'])
                + investing_flows['asset_sales']
                - abs(investing_flows['investment_purchases'])
                + investing_flows['investment_sales']
                + investing_flows['other_investing']
            )
            
            return investing_flows
            
        except Exception as e:
            self.logger.error(f"Error calculating investing cash flow: {str(e)}")
            raise

    def _get_asset_purchases(self,
                           gl_coa: pd.DataFrame,
                           gl_journals: pd.DataFrame,
                           gl_lines: pd.DataFrame,
                           fa_assets: pd.DataFrame) -> float:
        """Calculate cash used for asset purchases"""
        try:
            # Get fixed asset related accounts
            asset_accounts = gl_coa[
                (gl_coa['account_type'] == 'ASSET') &
                (gl_coa['account_category'] == 'FIXED')
            ]['account_id'].tolist()
            
            # Get asset purchase transactions
            purchase_entries = gl_lines[
                (gl_lines['account_id'].isin(asset_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            # Cross-reference with FA_ASSETS
            new_assets = fa_assets[
                fa_assets['acquisition_date'].between(
                    self.start_date, self.end_date
                )
            ]
            
            # Calculate purchases from both sources
            gl_purchases = purchase_entries['debit_amount'].sum()
            fa_purchases = new_assets['acquisition_cost'].sum()
            
            # Use the larger amount to ensure we capture all purchases
            total_purchases = max(gl_purchases, fa_purchases)
            
            return float(total_purchases or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating asset purchases: {str(e)}")
            return 0.0

    def _get_asset_sales(self,
                        gl_coa: pd.DataFrame,
                        gl_journals: pd.DataFrame,
                        gl_lines: pd.DataFrame,
                        fa_assets: pd.DataFrame) -> float:
        """Calculate cash received from asset sales"""
        try:
            # Get fixed asset related accounts
            asset_accounts = gl_coa[
                (gl_coa['account_type'] == 'ASSET') &
                (gl_coa['account_category'] == 'FIXED')
            ]['account_id'].tolist()
            
            # Get asset sale transactions
            sale_entries = gl_lines[
                (gl_lines['account_id'].isin(asset_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            # Get disposed assets from FA_ASSETS
            disposed_assets = fa_assets[
                (fa_assets['status'] == 'DISPOSED') &
                (pd.to_datetime(fa_assets['created_date']).between(
                    self.start_date, self.end_date
                ))
            ]
            
            # Calculate sales from journal entries
            gl_sales = sale_entries['credit_amount'].sum()
            
            # Add any disposal proceeds not captured in GL
            fa_disposals = disposed_assets['book_value'].sum()
            
            total_sales = gl_sales + fa_disposals
            
            return float(total_sales or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating asset sales: {str(e)}")
            return 0.0

    def _get_investment_purchases(self,
                                gl_coa: pd.DataFrame,
                                gl_journals: pd.DataFrame,
                                gl_lines: pd.DataFrame) -> float:
        """Calculate cash used for investment purchases"""
        try:
            # Get investment accounts
            investment_accounts = gl_coa[
                (gl_coa['account_type'] == 'ASSET') &
                (gl_coa['account_category'].str.contains('INVEST', case=False, na=False))
            ]['account_id'].tolist()
            
            # Get investment purchase transactions
            purchase_entries = gl_lines[
                (gl_lines['account_id'].isin(investment_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if purchase_entries.empty:
                return 0.0
                
            # Calculate net purchases
            purchases = purchase_entries['debit_amount'].sum()
            
            return float(purchases or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating investment purchases: {str(e)}")
            return 0.0
        


    def _get_investment_sales(self,
                            gl_coa: pd.DataFrame,
                            gl_journals: pd.DataFrame,
                            gl_lines: pd.DataFrame) -> float:
        """Calculate cash received from investment sales"""
        try:
            # Get investment accounts
            investment_accounts = gl_coa[
                (gl_coa['account_type'] == 'ASSET') &
                (gl_coa['account_category'].str.contains('INVEST', case=False, na=False))
            ]['account_id'].tolist()
            
            # Get investment sale transactions
            sale_entries = gl_lines[
                (gl_lines['account_id'].isin(investment_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if sale_entries.empty:
                return 0.0
                
            # Calculate investment sales
            sales = sale_entries['credit_amount'].sum()
            
            return float(sales or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating investment sales: {str(e)}")
            return 0.0

    def _get_other_investing_flows(self,
                                 gl_coa: pd.DataFrame,
                                 gl_journals: pd.DataFrame,
                                 gl_lines: pd.DataFrame) -> float:
        """Calculate other investing cash flows"""
        try:
            # Get other investing related accounts
            other_accounts = gl_coa[
                (gl_coa['account_type'].isin(['ASSET', 'LIABILITY'])) &
                (gl_coa['account_category'].str.contains('INVEST', case=False, na=False)) &
                (~gl_coa['account_category'].isin(['FIXED_ASSET', 'INVESTMENT']))
            ]['account_id'].tolist()
            
            # Get other investing transactions
            other_entries = gl_lines[
                (gl_lines['account_id'].isin(other_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if other_entries.empty:
                return 0.0
                
            # Calculate net other investing flows
            other_flows = other_entries['credit_amount'].sum() - other_entries['debit_amount'].sum()
            
            return float(other_flows)
            
        except Exception as e:
            self.logger.error(f"Error calculating other investing flows: {str(e)}")
            return 0.0

    def _calculate_financing_cash_flow(self) -> Dict[str, float]:
        """Calculate financing cash flow components"""
        try:
            # Get required data
            gl_coa = self.data_loader.data['gl_coa']
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            
            financing_flows = {
                'debt_proceeds': self._get_debt_proceeds(
                    gl_coa, gl_journals, gl_lines
                ),
                'debt_payments': self._get_debt_payments(
                    gl_coa, gl_journals, gl_lines
                ),
                'equity_proceeds': self._get_equity_proceeds(
                    gl_coa, gl_journals, gl_lines
                ),
                'dividends_paid': self._get_dividend_payments(
                    gl_coa, gl_journals, gl_lines
                ),
                'other_financing': self._get_other_financing_flows(
                    gl_coa, gl_journals, gl_lines
                )
            }
            
            # Calculate net financing cash flow
            financing_flows['net_cash'] = (
                financing_flows['debt_proceeds']
                - abs(financing_flows['debt_payments'])
                + financing_flows['equity_proceeds']
                - abs(financing_flows['dividends_paid'])
                + financing_flows['other_financing']
            )
            
            return financing_flows
            
        except Exception as e:
            self.logger.error(f"Error calculating financing cash flow: {str(e)}")
            raise

    def _get_debt_proceeds(self,
                          gl_coa: pd.DataFrame,
                          gl_journals: pd.DataFrame,
                          gl_lines: pd.DataFrame) -> float:
        """Calculate cash proceeds from debt"""
        try:
            # Get debt-related accounts
            debt_accounts = gl_coa[
                (gl_coa['account_type'] == 'LIABILITY') &
                (gl_coa['account_category'].str.contains('LOAN|DEBT|BORROWING', 
                                                       case=False, na=False))
            ]['account_id'].tolist()
            
            # Get debt proceed transactions
            proceed_entries = gl_lines[
                (gl_lines['account_id'].isin(debt_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if proceed_entries.empty:
                return 0.0
                
            # Calculate debt proceeds
            proceeds = proceed_entries['credit_amount'].sum()
            
            return float(proceeds or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating debt proceeds: {str(e)}")
            return 0.0
        

    

    def _get_debt_payments(self,
                          gl_coa: pd.DataFrame,
                          gl_journals: pd.DataFrame,
                          gl_lines: pd.DataFrame) -> float:
        """Calculate debt repayments"""
        try:
            # Get debt-related accounts
            debt_accounts = gl_coa[
                (gl_coa['account_type'] == 'LIABILITY') &
                (gl_coa['account_category'].str.contains('LOAN|DEBT|BORROWING', 
                                                       case=False, na=False))
            ]['account_id'].tolist()
            
            # Get debt payment transactions
            payment_entries = gl_lines[
                (gl_lines['account_id'].isin(debt_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if payment_entries.empty:
                return 0.0
                
            # Calculate debt payments
            payments = payment_entries['debit_amount'].sum()
            
            return float(payments or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating debt payments: {str(e)}")
            return 0.0

    def _get_equity_proceeds(self,
                           gl_coa: pd.DataFrame,
                           gl_journals: pd.DataFrame,
                           gl_lines: pd.DataFrame) -> float:
        """Calculate proceeds from equity issuance"""
        try:
            # Get equity capital accounts
            equity_accounts = gl_coa[
                (gl_coa['account_type'] == 'EQUITY') &
                (gl_coa['account_category'].str.contains('CAPITAL|SHARE', 
                                                       case=False, na=False))
            ]['account_id'].tolist()
            
            # Get equity issuance transactions
            equity_entries = gl_lines[
                (gl_lines['account_id'].isin(equity_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if equity_entries.empty:
                return 0.0
                
            # Calculate equity proceeds
            proceeds = equity_entries['credit_amount'].sum()
            
            return float(proceeds or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating equity proceeds: {str(e)}")
            return 0.0

    def _get_dividend_payments(self,
                             gl_coa: pd.DataFrame,
                             gl_journals: pd.DataFrame,
                             gl_lines: pd.DataFrame) -> float:
        """Calculate dividend payments"""
        try:
            # Get dividend-related accounts
            dividend_accounts = gl_coa[
                (gl_coa['account_type'] == 'EQUITY') &
                (gl_coa['account_category'].str.contains('DIVIDEND', 
                                                       case=False, na=False))
            ]['account_id'].tolist()
            
            # Get dividend payment transactions
            dividend_entries = gl_lines[
                (gl_lines['account_id'].isin(dividend_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if dividend_entries.empty:
                return 0.0
                
            # Calculate dividend payments
            payments = dividend_entries['debit_amount'].sum()
            
            return float(payments or 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating dividend payments: {str(e)}")
            return 0.0

    def _get_other_financing_flows(self,
                                 gl_coa: pd.DataFrame,
                                 gl_journals: pd.DataFrame,
                                 gl_lines: pd.DataFrame) -> float:
        """Calculate other financing cash flows"""
        try:
            # Get other financing accounts (excluding main categories)
            excluded_categories = ['LOAN', 'DEBT', 'BORROWING', 'CAPITAL', 'SHARE', 'DIVIDEND']
            other_accounts = gl_coa[
                (gl_coa['account_type'].isin(['LIABILITY', 'EQUITY'])) &
                (~gl_coa['account_category'].str.contains('|'.join(excluded_categories), 
                                                        case=False, na=False))
            ]['account_id'].tolist()
            
            # Get other financing transactions
            other_entries = gl_lines[
                (gl_lines['account_id'].isin(other_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'].between(
                            self.start_date, self.end_date
                        )) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if other_entries.empty:
                return 0.0
                
            # Calculate net other financing flows
            other_flows = other_entries['credit_amount'].sum() - other_entries['debit_amount'].sum()
            
            return float(other_flows)
            
        except Exception as e:
            self.logger.error(f"Error calculating other financing flows: {str(e)}")
            return 0.0

    def _get_cash_position(self, as_of_date: datetime) -> float:
        """Get cash and cash equivalent balance at a specific date"""
        try:
            gl_coa = self.data_loader.data['gl_coa']
            gl_journals = self.data_loader.data['gl_journals']
            gl_lines = self.data_loader.data['gl_journal_lines']
            
            # Get cash and equivalent accounts
            cash_accounts = gl_coa[
                gl_coa['account_type'] == 'ASSET'
            ][gl_coa['account_category'].isin(self.cash_account_types)]['account_id'].tolist()
            
            # Get balances for cash accounts
            cash_entries = gl_lines[
                (gl_lines['account_id'].isin(cash_accounts)) &
                (gl_lines['journal_id'].isin(
                    gl_journals[
                        (gl_journals['journal_date'] <= as_of_date) &
                        (gl_journals['status'] == 'POSTED')
                    ]['journal_id']
                ))
            ]
            
            if cash_entries.empty:
                return 0.0
                
            # Calculate net cash position
            cash_position = (
                cash_entries['debit_amount'].sum() - 
                cash_entries['credit_amount'].sum()
            )
            
            return float(cash_position if cash_position > 0 else 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating cash position: {str(e)}")
            return 0.0

    def _calculate_cash_flow_metrics(self,
                                   operating_cash_flow: Dict[str, float],
                                   investing_cash_flow: Dict[str, float],
                                   financing_cash_flow: Dict[str, float],
                                   starting_cash: float) -> Dict[str, float]:
        """Calculate key cash flow metrics and ratios"""
        try:
            metrics = {}
            
            # Operating cash flow metrics
            if starting_cash > 0:
                metrics['operating_cash_ratio'] = (
                    operating_cash_flow['net_cash'] / starting_cash
                )
            else:
                metrics['operating_cash_ratio'] = 0.0
                
            # Cash sufficiency metrics
            total_outflows = (
                abs(investing_cash_flow['asset_purchases']) +
                abs(investing_cash_flow['investment_purchases']) +
                abs(financing_cash_flow['debt_payments'])
            )
            
            if total_outflows > 0:
                metrics['cash_sufficiency_ratio'] = (
                    operating_cash_flow['net_cash'] / total_outflows
                )
            else:
                metrics['cash_sufficiency_ratio'] = 1.0
                
            # Cash reinvestment ratio
            if operating_cash_flow['net_cash'] > 0:
                metrics['reinvestment_ratio'] = (
                    abs(investing_cash_flow['asset_purchases']) /
                    operating_cash_flow['net_cash']
                )
            else:
                metrics['reinvestment_ratio'] = 0.0
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating cash flow metrics: {str(e)}")
            return {}
        
    

    def _analyze_trends(self, months: int = 12) -> Dict:
        """Analyze cash flow trends over time"""
        try:
            trend_start = self.end_date - timedelta(days=months*30)
            
            trends = {
                'operating_cash_flow': [],
                'investing_cash_flow': [],
                'financing_cash_flow': [],
                'net_cash_flow': [],
                'cash_position': []
            }
            
            # Calculate metrics for each month
            current_date = trend_start
            while current_date <= self.end_date:
                # Set analysis period to current month
                self.start_date = current_date.replace(day=1)
                self.end_date = (current_date.replace(day=1) + 
                               timedelta(days=32)).replace(day=1) - timedelta(days=1)
                
                # Calculate monthly flows
                operating = self._calculate_operating_cash_flow()
                investing = self._calculate_investing_cash_flow()
                financing = self._calculate_financing_cash_flow()
                
                # Calculate net flow
                net_flow = (
                    operating['net_cash'] +
                    investing['net_cash'] +
                    financing['net_cash']
                )
                
                # Get end of month cash position
                cash_position = self._get_cash_position(self.end_date)
                
                # Store monthly values
                trends['operating_cash_flow'].append({
                    'date': current_date,
                    'value': operating['net_cash']
                })
                
                trends['investing_cash_flow'].append({
                    'date': current_date,
                    'value': investing['net_cash']
                })
                
                trends['financing_cash_flow'].append({
                    'date': current_date,
                    'value': financing['net_cash']
                })
                
                trends['net_cash_flow'].append({
                    'date': current_date,
                    'value': net_flow
                })
                
                trends['cash_position'].append({
                    'date': current_date,
                    'value': cash_position
                })
                
                # Move to next month
                current_date = (current_date.replace(day=1) + 
                              timedelta(days=32)).replace(day=1)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {str(e)}")
            raise

    def get_cash_flow_summary(self) -> Dict:
        """Get formatted cash flow summary"""
        if not self.results:
            raise ValueError("No analysis results available. Run analyze_cash_flow() first.")
            
        return {
            'period': {
                'start_date': self.results['period']['start_date'].strftime('%Y-%m-%d'),
                'end_date': self.results['period']['end_date'].strftime('%Y-%m-%d')
            },
            'cash_positions': {
                'starting_balance': self.results['cash_positions']['starting_cash'],
                'ending_balance': self.results['cash_positions']['ending_cash'],
                'net_change': self.results['cash_positions']['net_change']
            },
            'cash_flows': {
                'operating': {
                    'net_cash': self.results['cash_flows']['operating']['net_cash'],
                    'details': {
                        k: v for k, v in self.results['cash_flows']['operating'].items()
                        if k != 'net_cash'
                    }
                },
                'investing': {
                    'net_cash': self.results['cash_flows']['investing']['net_cash'],
                    'details': {
                        k: v for k, v in self.results['cash_flows']['investing'].items()
                        if k != 'net_cash'
                    }
                },
                'financing': {
                    'net_cash': self.results['cash_flows']['financing']['net_cash'],
                    'details': {
                        k: v for k, v in self.results['cash_flows']['financing'].items()
                        if k != 'net_cash'
                    }
                },
                'total_net_cash_flow': self.results['cash_flows']['net_cash_flow']
            },
            'key_metrics': self.results['metrics']
        }