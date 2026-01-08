# config/database_config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '1521'),
    'service_name': os.getenv('DB_SERVICE_NAME', 'ORCL'),
    'user': os.getenv('DB_USER', 'your_username'),
    'password': os.getenv('DB_PASSWORD', 'your_password')
}

# Define SQL queries for main tables
QUERIES = {
    'gl_coa': '''
        SELECT * FROM GL_COA 
        WHERE is_active = 'Y'
    ''',
    'gl_journals': '''
        SELECT * FROM GL_JOURNALS 
        WHERE journal_date >= ADD_MONTHS(SYSDATE, -12)
    ''',
    'gl_journal_lines': '''
        SELECT * FROM GL_JOURNAL_LINES
    ''',
    'ap_invoices': '''
        SELECT * FROM AP_INVOICES 
        WHERE invoice_date >= ADD_MONTHS(SYSDATE, -12)
    ''',
    'ap_payments': '''
        SELECT * FROM AP_PAYMENTS
        WHERE payment_date >= ADD_MONTHS(SYSDATE, -12)
    ''',
    'fa_assets': '''
        SELECT * FROM FA_ASSETS
    ''',
    'fa_depreciation': '''
        SELECT * FROM FA_DEPRECIATION 
        WHERE period_date >= ADD_MONTHS(SYSDATE, -12)
    '''
}

VALIDATION_QUERIES = {
    'check_tables': '''
        SELECT table_name 
        FROM user_tables 
        WHERE table_name IN (
            'GL_COA',
            'GL_JOURNALS',
            'GL_JOURNAL_LINES',
            'AP_INVOICES',
            'AP_PAYMENTS',
            'FA_ASSETS',
            'FA_DEPRECIATION'
        )
    ''',
    'check_permissions': '''
        SELECT privilege 
        FROM user_tab_privs 
        WHERE table_name IN (
            'GL_COA',
            'GL_JOURNALS',
            'GL_JOURNAL_LINES',
            'AP_INVOICES',
            'AP_PAYMENTS',
            'FA_ASSETS',
            'FA_DEPRECIATION'
        )
    '''
}