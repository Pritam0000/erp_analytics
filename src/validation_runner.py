import sys
sys.path.append(r"C:/Users/khush/Downloads/projects/Oracle_ERP_Financial_Analytics_Enhancement")
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import json
from src.utils.database_utils import DatabaseConnector
from src.data_loading.data_loader import FinancialDataLoader
from src.data_validation.streamlined_validator import StreamlinedValidator

def setup_logging():
    """Setup logging configuration with both file and console handlers"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp-based log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'validation_run_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for reports and logs"""
    directories = ['logs', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
def debug_print_data_info(data, logger):
    """Print debugging information about loaded data"""
    logger.info("\n=== Data Loading Summary ===")
    for table_name, df in data.items():
        logger.info(f"\nTable: {table_name}")
        logger.info(f"Rows: {len(df)}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Print sample data for key tables
        if table_name in ['gl_journal_lines', 'gl_journals']:
            logger.info(f"\nSample data for {table_name}:")
            logger.info(df.head().to_string())
            
        # Check for null values in key columns
        if not df.empty:
            null_counts = df.isnull().sum()
            if any(null_counts > 0):
                logger.info(f"\nNull value counts in {table_name}:")
                logger.info(null_counts[null_counts > 0].to_string())

def run_validation():
    """Run the validation process with enhanced error handling and logging"""
    logger = setup_logging()
    create_directories()
    
    try:
        logger.info("Starting validation process")
        
        # 1. Load financial data
        logger.info("Loading financial data")
        loader = FinancialDataLoader()
        data = loader.load_financial_data()
        
        # Print debugging information about loaded data
        debug_print_data_info(data, logger)
        
        # 2. Run validations
        logger.info("Running validations")
        validator = StreamlinedValidator(data)
        
        try:
            validation_results = validator.run_critical_validations()
            logger.info("Validation completed")
        except Exception as validation_error:
            logger.error(f"Validation failed: {str(validation_error)}", exc_info=True)
            raise
        
        # 3. Generate and save report
        report = validator.get_validation_report()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/validation_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
        
        # 4. Print summary to console
        logger.info("\nValidation Summary:")
        logger.info(f"Status: {report['validation_status']}")
        logger.info(f"Critical Issues: {report['critical_issues']}")
        logger.info(f"Warnings: {report['warnings']}")
        
        # Return True if validation passed
        return report['validation_status'] == 'PASSED'
        
    except Exception as e:
        logger.error("Validation process failed", exc_info=True)
        raise
    finally:
        # Add any cleanup code here if needed
        logger.info("Validation process completed")
        
def main():
    try:
        validation_passed = run_validation()
        exit_code = 0 if validation_passed else 1
        print(f"\nValidation {'passed' if validation_passed else 'failed'}")
        exit(exit_code)
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()