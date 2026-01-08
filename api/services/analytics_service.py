# api/services/analytics_service.py

from src.data_loading.data_loader import FinancialDataLoader
from src.analytics.financial_analysis.balance_sheet_analyzer import BalanceSheetAnalyzer
from src.analytics.financial_analysis.income_statement_analyzer import IncomeStatementAnalyzer
from src.analytics.financial_analysis.cash_flow_analyzer import CashFlowAnalyzer
from src.analytics.financial_analysis.ratio_analyzer import RatioAnalyzer

class AnalyticsService:
    """Service class for financial analytics"""
    
    def __init__(self):
        self.data_loader = FinancialDataLoader()
        self.balance_sheet_analyzer = BalanceSheetAnalyzer(self.data_loader)
        self.income_statement_analyzer = IncomeStatementAnalyzer(self.data_loader)
        self.cash_flow_analyzer = CashFlowAnalyzer(self.data_loader)
        self.ratio_analyzer = RatioAnalyzer(self.data_loader)
    
    def get_balance_sheet_analysis(self, as_of_date=None):
        """Get balance sheet analysis"""
        analysis = self.balance_sheet_analyzer.analyze_balance_sheet(as_of_date)
        return self.balance_sheet_analyzer.get_balance_sheet_summary()
    
    def get_income_statement_analysis(self, start_date=None, end_date=None):
        """Get income statement analysis"""
        analysis = self.income_statement_analyzer.analyze_income_statement(
            start_date, end_date
        )
        return self.income_statement_analyzer.get_income_statement_summary()
    
    def get_cash_flow_analysis(self, start_date=None, end_date=None):
        """Get cash flow analysis"""
        analysis = self.cash_flow_analyzer.analyze_cash_flow(
            start_date, end_date
        )
        return self.cash_flow_analyzer.get_cash_flow_summary()
    
    def get_financial_ratios(self, as_of_date=None):
        """Get financial ratio analysis"""
        analysis = self.ratio_analyzer.calculate_ratios(as_of_date)
        return self.ratio_analyzer.get_ratio_summary()