# src/utils/mock_database.py

import pandas as pd
import logging
from datetime import datetime, timedelta
import numpy as np

class MockDatabaseConnector:
    """Mock database connector for development without Oracle connection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_mock_data()
        
    def _initialize_mock_data(self):
        """Initialize mock data tables"""
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
        self.gl_journal_lines = pd.DataFrame({
            'line_id': range(1, 21),
            'journal_id': [i for i in range(1, 11) for _ in range(2)],
            'account_id': np.random.choice(range(1001, 1011), 20),
            'debit_amount': np.random.uniform(100, 1000, 20),
            'credit_amount': np.random.uniform(100, 1000, 20),
            'created_date': [datetime.now() - timedelta(days=x) for x in range(20)]
        })

    def execute_query(self, query: str, params=None) -> pd.DataFrame:
        """Execute mock query and return appropriate mock data"""
        try:
            # Log the query for debugging
            self.logger.debug(f"Mock executing query: {query}")
            
            # Return appropriate mock data based on query content
            if 'GL_COA' in query:
                return self.gl_coa
            elif 'GL_JOURNALS' in query:
                return self.gl_journals
            elif 'GL_JOURNAL_LINES' in query:
                return self.gl_journal_lines
            else:
                return pd.DataFrame()  # Empty DataFrame for unknown queries
                
        except Exception as e:
            self.logger.error(f"Error executing mock query: {str(e)}")
            raise

    def validate_connection(self) -> dict:
        """Mock connection validation"""
        return {
            'connection_ok': True,
            'tables_exist': True,
            'permissions_ok': True
        }

    def close(self):
        """Mock close connection"""
        self.logger.info("Mock database connection closed")