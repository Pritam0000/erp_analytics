import pandas as pd 
import numpy as np
import logging
from datetime import datetime
from src.utils.database_utils import DatabaseConnector
from config.database_config import QUERIES

class FinancialDataLoader:
    def __init__(self):
        self.db = DatabaseConnector()
        self.data = {}
        self.load_timestamp = None
        self.logger = logging.getLogger(__name__)

    def load_financial_data(self):
        """Load all required financial data tables"""
        try:
            self.db.connect()
            self.load_timestamp = datetime.now()

            # Load main tables with error handling for each
            tables_to_load = {
                'gl_coa': self._load_chart_of_accounts,
                'ap_suppliers': self._load_suppliers,
                'ap_invoices': self._load_invoices,
                'fa_categories': self._load_fa_categories,
                'gl_journals': self._load_journals,
                'fa_assets': self._load_fa_assets,
                'ap_invoice_lines': self._load_ap_invoice_lines,
                'gl_journal_lines': self._load_journal_lines,
                'fa_depreciation': self._load_fa_depreciation,
                'ap_payments': self._load_ap_payments
            }

            for table_name, load_function in tables_to_load.items():
                try:
                    self.logger.info(f"Loading {table_name}")
                    self.data[table_name] = load_function()
                except Exception as e:
                    self.logger.error(f"Error loading {table_name}: {str(e)}")
                    # Initialize empty DataFrame with correct columns if load fails
                    self.data[table_name] = pd.DataFrame()
                    continue

            return self.data

        except Exception as e:
            self.logger.error(f"Error loading financial data: {str(e)}")
            raise
        finally:
            self.db.disconnect()

    def _safe_convert_to_datetime(self, df, date_columns):
        """Safely convert date columns to datetime"""
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Error converting {col} to datetime: {str(e)}")
        return df

    def _load_chart_of_accounts(self):
        """Load and preprocess Chart of Accounts"""
        try:
            df = self.db.execute_query("""
                SELECT 
                    account_id,
                    account_number,
                    account_name,
                    account_type,
                    account_category,
                    parent_account_id,
                    is_active,
                    created_date,
                    created_by,
                    last_updated_date,
                    last_updated_by
                FROM GL_COA
            """)
            date_columns = ['created_date', 'last_updated_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_chart_of_accounts: {str(e)}")
            raise

    def _load_suppliers(self):
        """Load and preprocess Suppliers"""
        try:
            df = self.db.execute_query("""
                SELECT
                    supplier_id,
                    supplier_number,
                    supplier_name,
                    tax_id,
                    payment_terms,
                    status,
                    created_date,
                    created_by
                FROM AP_SUPPLIERS
            """)
            date_columns = ['created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_suppliers: {str(e)}")
            raise

    def _load_invoices(self):
        """Load and preprocess Invoices"""
        try:
            df = self.db.execute_query("""
                SELECT
                    invoice_id,
                    supplier_id,
                    invoice_number,
                    invoice_date,
                    due_date,
                    amount,
                    status,
                    created_date,
                    created_by
                FROM AP_INVOICES
            """)
            date_columns = ['invoice_date', 'due_date', 'created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_invoices: {str(e)}")
            raise

    def _load_fa_categories(self):
        """Load and preprocess Fixed Asset Categories"""
        try:
            df = self.db.execute_query("""
                SELECT
                    category_id,
                    category_code,
                    category_name,
                    depreciation_method,
                    life_years,
                    created_date,
                    created_by
                FROM FA_CATEGORIES
            """)
            date_columns = ['created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_fa_categories: {str(e)}")
            raise

    def _load_journals(self):
        """Load and preprocess GL Journals"""
        try:
            df = self.db.execute_query("""
                SELECT
                    journal_id,
                    journal_date,
                    journal_number,
                    batch_number,
                    description,
                    status,
                    period_name,
                    created_by,
                    created_date,
                    posted_by,
                    posted_date
                FROM GL_JOURNALS
            """)
            date_columns = ['journal_date', 'created_date', 'posted_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_journals: {str(e)}")
            raise

    def _load_fa_assets(self):
        """Load and preprocess Fixed Assets"""
        try:
            df = self.db.execute_query("""
                SELECT
                    asset_id,
                    asset_number,
                    asset_name,
                    category_id,
                    acquisition_date,
                    acquisition_cost,
                    salvage_value,
                    life_years,
                    status,
                    created_date,
                    created_by
                FROM FA_ASSETS
            """)
            date_columns = ['acquisition_date', 'created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_fa_assets: {str(e)}")
            raise

    def _load_ap_invoice_lines(self):
        """Load and preprocess AP Invoice Lines"""
        try:
            df = self.db.execute_query("""
                SELECT
                    line_id,
                    invoice_id,
                    account_id,
                    description,
                    amount,
                    tax_amount,
                    created_date,
                    created_by
                FROM AP_INVOICE_LINES
            """)
            date_columns = ['created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_ap_invoice_lines: {str(e)}")
            raise

    def _load_journal_lines(self):
        """Load and preprocess GL Journal Lines"""
        try:
            df = self.db.execute_query("""
                SELECT  
                    line_id,
                    journal_id, 
                    line_number,
                    account_id,
                    debit_amount,
                    credit_amount,
                    description,
                    created_by,
                    created_date
                FROM GL_JOURNAL_LINES
            """)
            date_columns = ['created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_journal_lines: {str(e)}")
            raise
    
    def _load_fa_depreciation(self):
        """Load and preprocess FA Depreciation"""
        try:
            df = self.db.execute_query("""
                SELECT
                    depreciation_id,
                    asset_id,
                    period_date,
                    depreciation_amount,
                    accumulated_depreciation,
                    book_value,
                    created_date,
                    created_by
                FROM FA_DEPRECIATION
            """)
            date_columns = ['period_date', 'created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_fa_depreciation: {str(e)}")
            raise

    def _load_ap_payments(self):
        """Load and preprocess AP Payments"""
        try:
            df = self.db.execute_query("""
                SELECT
                    payment_id,
                    supplier_id,
                    payment_number,
                    payment_date,
                    payment_method,
                    amount,
                    status,
                    created_date,
                    created_by
                FROM AP_PAYMENTS
            """)
            date_columns = ['payment_date', 'created_date']
            df = self._safe_convert_to_datetime(df, date_columns)
            return df
        except Exception as e:
            self.logger.error(f"Error in _load_ap_payments: {str(e)}")
            raise

    def get_data_summary(self):
        """Generate summary of loaded data"""
        try:
            summary = {}
            for table_name, df in self.data.items():
                if df.empty:
                    summary[table_name] = {
                        'status': 'EMPTY',
                        'error': 'Table failed to load'
                    }
                    continue
                    
                summary[table_name] = {
                    'status': 'LOADED',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                    'date_range': self._get_date_range(df),
                    'null_counts': df.isnull().sum().to_dict()
                }
            return summary
        except Exception as e:
            self.logger.error(f"Error generating data summary: {str(e)}")
            return {'error': str(e)}
            
    def _get_date_range(self, df):
        """Get date range for tables with date columns"""
        try:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                return {
                    col: {
                        'min': df[col].min(),
                        'max': df[col].max()    
                    } for col in date_cols
                }
            return None
        except Exception as e:
            self.logger.warning(f"Error getting date range: {str(e)}")
            return None