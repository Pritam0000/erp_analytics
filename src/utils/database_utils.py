# src/utils/database_utils.py

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class DatabaseConnector:
    def __init__(self):
        """Initialize database connector with mock data by default"""
        self.logger = logging.getLogger(__name__)
        self.use_mock = os.getenv('USE_MOCK_DB', 'true').lower() == 'true'
        
        if self.use_mock:
            self.logger.info("Using mock database")
            self._init_mock_data()
        else:
            self.logger.info("Attempting to connect to Oracle")
            try:
                self._init_connection_pool()
            except Exception as e:
                self.logger.error(f"Error initializing connection pool: {str(e)}")
                self.logger.info("Falling back to mock data")
                self.use_mock = True
                self._init_mock_data()

    def _init_mock_data(self):
        """Initialize mock data"""
        # Mock GL Chart of Accounts
        self.gl_coa = pd.DataFrame({
            'account_id': range(1001, 1011),
            'account_number': [f'AC{i}' for i in range(1001, 1011)],
            'account_name': [f'Account {i}' for i in range(1001, 1011)],
            'account_type': ['ASSET', 'LIABILITY', 'EQUITY', 'REVENUE', 'EXPENSE'] * 2,
            'account_category': ['CURRENT', 'FIXED', 'OPERATING', 'FINANCIAL', 'GENERAL'] * 2,
            'is_active': ['Y'] * 10,
            'created_date': [datetime.now() - timedelta(days=x) for x in range(10)]
        })
        
        # Mock GL Journals
        self.gl_journals = pd.DataFrame({
            'journal_id': range(1, 11),
            'journal_date': [datetime.now() - timedelta(days=x) for x in range(10)],
            'journal_number': [f'JE{i:04d}' for i in range(1, 11)],
            'status': ['POSTED'] * 10,
            'created_date': [datetime.now() - timedelta(days=x) for x in range(10)]
        })
        
        # Mock Journal Lines
        self.gl_lines = pd.DataFrame({
            'line_id': range(1, 21),
            'journal_id': [i for i in range(1, 11) for _ in range(2)],
            'account_id': np.random.choice(range(1001, 1011), 20),
            'debit_amount': np.random.uniform(1000, 10000, 20),
            'credit_amount': np.random.uniform(1000, 10000, 20),
            'created_date': [datetime.now() - timedelta(days=x) for x in range(20)]
        })

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute query using either mock or real database"""
        try:
            if self.use_mock:
                return self._execute_mock_query(query, params)
            else:
                return self._execute_real_query(query, params)
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def _execute_mock_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute query against mock data"""
        try:
            # Basic query parsing to return appropriate mock data
            query = query.upper()
            
            if 'GL_COA' in query:
                df = self.gl_coa.copy()
            elif 'GL_JOURNALS' in query:
                df = self.gl_journals.copy()
            elif 'GL_JOURNAL_LINES' in query:
                df = self.gl_lines.copy()
            else:
                return pd.DataFrame()  # Empty DataFrame for unknown tables
                
            # Apply basic filtering if parameters provided
            if params:
                for key, value in params.items():
                    if key in ['start_date', 'from_date']:
                        df = df[df['journal_date'] >= pd.to_datetime(value)]
                    elif key in ['end_date', 'to_date']:
                        df = df[df['journal_date'] <= pd.to_datetime(value)]
                    elif key in ['status', 'account_type']:
                        df = df[df[key.lower()] == value]
                        
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing mock query: {str(e)}")
            raise

    def _execute_real_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute query against real Oracle database"""
        try:
            import cx_Oracle
            return pd.read_sql(query, self.connection, params=params)
        except Exception as e:
            self.logger.error(f"Error executing real query: {str(e)}")
            raise

    def _init_connection_pool(self):
        """Initialize Oracle connection pool"""
        try:
            import cx_Oracle
            # Oracle connection code here (if needed in future)
            pass
        except Exception as e:
            self.logger.error(f"Error initializing DatabaseConnector: {str(e)}")
            raise