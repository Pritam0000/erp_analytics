# streamlined_validator.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

class StreamlinedValidator:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        # Make a copy of the data with lowercase column names
        self.data = {
            table: df.copy().rename(columns=str.lower) 
            for table, df in data.items()
        }
        self.logger = self.setup_logging()
        self.validation_summary = {
            'timestamp': datetime.now(),
            'critical_issues': [],
            'warnings': [],
            'validation_status': True
        }

    def setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Check if logger already has handlers to avoid duplicate handlers
        if not logger.handlers:
            # Create logs directory if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Create a timestamp-based log filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'validation_{timestamp}.log'
            
            # Add file handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            # Add console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
        return logger

    def run_critical_validations(self) -> Dict[str, Any]:
        """Run all critical validations in sequence"""
        try:
            self.logger.info("Starting critical validations")
            
            # Validate data presence
            self._validate_data_presence()
            
            # Only proceed with other validations if we have data
            if self.validation_summary['validation_status']:
                # 1. Financial Balance Checks
                self._validate_journal_balances()
                
                # 2. Data Integrity Checks
                self._validate_data_integrity()
                
                # 3. Reference Checks
                self._validate_references()
            
            # Generate summary report
            self._generate_validation_summary()
            
            return self.validation_summary
            
        except Exception as e:
            self.logger.error(f"Validation process failed: {str(e)}", exc_info=True)
            raise

    def _validate_data_presence(self) -> None:
        """Validate that required data is present"""
        required_tables = ['gl_journals', 'gl_journal_lines', 'gl_coa']
        missing_tables = []
        empty_tables = []
        
        for table in required_tables:
            if table not in self.data:
                missing_tables.append(table)
            elif self.data[table].empty:
                empty_tables.append(table)
        
        if missing_tables or empty_tables:
            self.validation_summary['critical_issues'].append({
                'type': 'DATA_AVAILABILITY',
                'missing_tables': missing_tables,
                'empty_tables': empty_tables
            })
            self.validation_summary['validation_status'] = False
            self.logger.error(f"Missing tables: {missing_tables}, Empty tables: {empty_tables}")

    def _validate_journal_balances(self):
        """Validate journal entry balances with case-insensitive column names"""
        try:
            if 'gl_journal_lines' not in self.data or self.data['gl_journal_lines'].empty:
                self.logger.warning("GL journal lines data is not available")
                return
                
            journal_lines = self.data['gl_journal_lines']
            
            # Log the structure of the dataframe
            self.logger.info(f"GL Journal Lines columns: {journal_lines.columns.tolist()}")
            self.logger.info(f"Sample data:\n{journal_lines.head()}")
            
            # Check if required columns exist (now case-insensitive)
            required_columns = ['journal_id', 'debit_amount', 'credit_amount']
            actual_columns = journal_lines.columns.str.lower()
            missing_columns = [col for col in required_columns if col not in actual_columns]
            
            if missing_columns:
                error_msg = f"Missing required columns in gl_journal_lines: {missing_columns}"
                self.logger.error(error_msg)
                self.validation_summary['critical_issues'].append({
                    'type': 'MISSING_COLUMNS',
                    'table': 'gl_journal_lines',
                    'missing_columns': missing_columns
                })
                return
            
            # Convert amount columns to numeric, replacing any non-numeric values with NaN
            journal_lines['debit_amount'] = pd.to_numeric(journal_lines['debit_amount'], errors='coerce')
            journal_lines['credit_amount'] = pd.to_numeric(journal_lines['credit_amount'], errors='coerce')

            # Calculate balances by journal_id
            balance_check = journal_lines.groupby('journal_id').agg({
                'debit_amount': 'sum',
                'credit_amount': 'sum'
            }).fillna(0)
            
            balance_check['difference'] = (
                balance_check['debit_amount'] - balance_check['credit_amount']
            ).abs()
            
            # Use a small tolerance for floating point comparisons
            tolerance = 0.01
            unbalanced = balance_check[balance_check['difference'] > tolerance]
            
            if not unbalanced.empty:
                self.logger.warning(f"Found {len(unbalanced)} unbalanced journals")
                self.validation_summary['critical_issues'].append({
                    'type': 'UNBALANCED_JOURNALS',
                    'count': len(unbalanced),
                    'journal_ids': unbalanced.index.tolist()[:10],  # List first 10 for reference
                    'max_difference': float(unbalanced['difference'].max())
                })
                self.validation_summary['validation_status'] = False
                
        except Exception as e:
            self.logger.error(f"Error in journal balance validation: {str(e)}", exc_info=True)
            raise

    def _validate_data_integrity(self) -> None:
        """Validate data integrity rules"""
        try:
            critical_checks = {
                'gl_journals': ['journal_id', 'journal_date', 'status'],
                'ap_invoices': ['invoice_id', 'supplier_id', 'amount'],
                'ap_payments': ['payment_id', 'supplier_id', 'amount']
            }

            for table, fields in critical_checks.items():
                if table not in self.data or self.data[table].empty:
                    self.logger.warning(f"Table {table} is not available for validation")
                    continue
                    
                df = self.data[table]
                
                # Check for required columns
                missing_columns = [col for col in fields if col not in df.columns]
                if missing_columns:
                    self.validation_summary['critical_issues'].append({
                        'type': 'MISSING_COLUMNS',
                        'table': table,
                        'missing_columns': missing_columns
                    })
                    continue
                
                # Check for null values in required fields
                null_checks = df[fields].isnull()
                if null_checks.any().any():
                    null_counts = null_checks.sum()
                    self.validation_summary['critical_issues'].append({
                        'type': 'NULL_VALUES',
                        'table': table,
                        'null_counts': null_counts[null_counts > 0].to_dict()
                    })
                    self.validation_summary['validation_status'] = False

        except Exception as e:
            self.logger.error(f"Error in data integrity validation: {str(e)}", exc_info=True)
            raise

    def _validate_references(self) -> None:
        """Validate referential integrity"""
        try:
            reference_checks = [
                ('ap_invoices', 'supplier_id', 'ap_suppliers', 'supplier_id'),
                ('gl_journal_lines', 'journal_id', 'gl_journals', 'journal_id'),
                ('gl_journal_lines', 'account_id', 'gl_coa', 'account_id')
            ]

            for child_table, child_key, parent_table, parent_key in reference_checks:
                if not all(table in self.data for table in [child_table, parent_table]):
                    self.logger.warning(f"Missing tables for reference check: {child_table} -> {parent_table}")
                    continue
                    
                if self.data[child_table].empty or self.data[parent_table].empty:
                    self.logger.warning(f"Empty tables for reference check: {child_table} -> {parent_table}")
                    continue

                # Check for required columns
                if not all(col in self.data[child_table].columns for col in [child_key]):
                    self.logger.error(f"Missing column {child_key} in {child_table}")
                    continue
                    
                if not all(col in self.data[parent_table].columns for col in [parent_key]):
                    self.logger.error(f"Missing column {parent_key} in {parent_table}")
                    continue

                # Check referential integrity
                child_keys = set(self.data[child_table][child_key].dropna())
                parent_keys = set(self.data[parent_table][parent_key].dropna())
                orphaned = child_keys - parent_keys

                if orphaned:
                    self.logger.warning(f"Found {len(orphaned)} orphaned records in {child_table}")
                    self.validation_summary['critical_issues'].append({
                        'type': 'REFERENTIAL_INTEGRITY',
                        'tables': f'{child_table} -> {parent_table}',
                        'count': len(orphaned),
                        'orphaned_keys': list(orphaned)[:10]  # List first 10 for reference
                    })
                    self.validation_summary['validation_status'] = False

        except Exception as e:
            self.logger.error(f"Error in reference validation: {str(e)}", exc_info=True)
            raise

    def _generate_validation_summary(self) -> None:
        """Generate final validation summary"""
        try:
            self.validation_summary['summary_stats'] = {
                'total_checks': 3,  # Number of main validation categories
                'failed_checks': len(self.validation_summary['critical_issues']),
                'warnings_count': len(self.validation_summary['warnings'])
            }
            
            # Log summary statistics
            self.logger.info("\nValidation Summary:")
            self.logger.info(f"Total Checks: {self.validation_summary['summary_stats']['total_checks']}")
            self.logger.info(f"Failed Checks: {self.validation_summary['summary_stats']['failed_checks']}")
            self.logger.info(f"Warnings: {self.validation_summary['summary_stats']['warnings_count']}")
            
        except Exception as e:
            self.logger.error(f"Error generating validation summary: {str(e)}", exc_info=True)
            raise

    def get_validation_report(self) -> Dict[str, Any]:
        """Get formatted validation report"""
        try:
            report = {
                'validation_timestamp': self.validation_summary['timestamp'],
                'validation_status': 'PASSED' if self.validation_summary['validation_status'] else 'FAILED',
                'critical_issues': len(self.validation_summary['critical_issues']),
                'warnings': len(self.validation_summary['warnings']),
                'details': {
                    'critical_issues': self.validation_summary['critical_issues'],
                    'warnings': self.validation_summary['warnings']
                }
            }
            return report
        except Exception as e:
            self.logger.error(f"Error generating validation report: {str(e)}", exc_info=True)
            raise