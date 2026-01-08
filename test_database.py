# test_database.py
from src.utils.database_utils import DatabaseConnector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection and validation"""
    try:
        # Initialize database connector
        logger.info("Initializing database connector...")
        db = DatabaseConnector()
        
        # Test connection validation
        logger.info("Testing connection validation...")
        validation = db.validate_connection()
        
        # Print validation results
        logger.info("\nValidation Results:")
        logger.info(f"Connection Status: {validation['connection']}")
        logger.info(f"Tables Exist: {validation['tables_exist']}")
        logger.info(f"Permissions OK: {validation['permissions']}")
        
        if validation['details'].get('missing_tables'):
            logger.warning(f"Missing tables: {validation['details']['missing_tables']}")
            
        # Test query execution
        if validation['connection'] and validation['tables_exist']:
            logger.info("\nTesting query execution...")
            
            # Test GL_COA query
            logger.info("Testing GL_COA query...")
            coa_data = db.execute_query("SELECT * FROM GL_COA WHERE ROWNUM <= 5")
            logger.info(f"Retrieved {len(coa_data)} rows from GL_COA")
            
            # Test GL_JOURNALS query
            logger.info("Testing GL_JOURNALS query...")
            journal_data = db.execute_query("SELECT * FROM GL_JOURNALS WHERE ROWNUM <= 5")
            logger.info(f"Retrieved {len(journal_data)} rows from GL_JOURNALS")
            
        return validation
        
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        raise
    finally:
        try:
            db.disconnect()
            logger.info("Database connection closed")
        except:
            pass

if __name__ == "__main__":
    print("Starting database connection test...")
    test_database_connection()