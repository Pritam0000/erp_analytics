# src/data_validation/validator.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class FinancialDataValidator:
    def __init__(self, data_dict):
        self.data = data_dict
        self.validation_results = {}
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('validation.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def validate_all(self):
        """Run all validations"""
        try:
            self.validate_data_completeness()
            self.validate_data_accuracy()
            self.validate_data_consistency()
            self.validate_business_rules()
            return self.validation_results
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

    def validate_data_completeness(self):
        """Check for missing and required data"""
        results = {}
        for table_name, df in self.data.items():
            results[table_name] = {
                'missing_values': {
                    col: df[col].isnull().sum() 
                    for col in df.columns
                },
                'total_rows': len(df),
                'complete_rows': len(df.dropna())
            }
        self.validation_results['completeness'] = results
        return results

    def validate_data_accuracy(self):
        """Validate data accuracy and format"""
        results = {}
        
        # Validate GL Journal balances
        if 'gl_journal_lines' in self.data:
            results['journal_balance'] = self._validate_journal_balances()
            
        # Validate payment amounts
        if 'ap_payments' in self.data:
            results['payment_accuracy'] = self._validate_payment_amounts()
            
        # Validate date ranges
        results['date_validations'] = self._validate_date_ranges()
        
        self.validation_results['accuracy'] = results
        return results

    def _validate_journal_balances(self):
        """Check if journals are balanced (debits = credits)"""
        journal_lines = self.data['gl_journal_lines']
        balances = journal_lines.groupby('journal_id').agg({
            'debit_amount': 'sum',
            'credit_amount': 'sum'
        })
        
        tolerance = 0.01  # Small tolerance for rounding differences
        balances['difference'] = abs(balances['debit_amount'] - balances['credit_amount'])
        unbalanced = balances[balances['difference'] > tolerance]
        
        return {
            'total_journals': len(balances),
            'unbalanced_count': len(unbalanced),
            'max_difference': balances['difference'].max(),
            'unbalanced_journals': unbalanced.index.tolist()
        }

    def validate_data_consistency(self):
        """Check for data consistency across tables"""
        results = {}
        
        # Check referential integrity
        if all(table in self.data for table in ['gl_journals', 'gl_journal_lines']):
            results['journal_integrity'] = self._check_referential_integrity(
                'gl_journal_lines', 'gl_journals', 'journal_id'
            )
            
        if all(table in self.data for table in ['ap_invoices', 'ap_suppliers']):
            results['supplier_integrity'] = self._check_referential_integrity(
                'ap_invoices', 'ap_suppliers', 'supplier_id'
            )
            
        self.validation_results['consistency'] = results
        return results

    def _check_referential_integrity(self, child_table, parent_table, key_field):
        """Check referential integrity between two tables"""
        child_keys = set(self.data[child_table][key_field])
        parent_keys = set(self.data[parent_table][key_field])
        orphaned_keys = child_keys - parent_keys
        
        return {
            'valid': len(orphaned_keys) == 0,
            'orphaned_records': len(orphaned_keys),
            'orphaned_keys': list(orphaned_keys)
        }

    def validate_business_rules(self):
        """Validate specific business rules"""
        results = {}
        
        # Validate account hierarchies
        if 'gl_coa' in self.data:
            results['account_hierarchy'] = self._validate_account_hierarchy()
            
        # Validate payment terms
        if 'ap_invoices' in self.data:
            results['payment_terms'] = self._validate_payment_terms()
            
        self.validation_results['business_rules'] = results
        return results

    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        report = {
            'timestamp': datetime.now(),
            'summary': {
                'tables_validated': list(self.data.keys()),
                'total_validations': len(self.validation_results)
            },
            'results': self.validation_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report

    def _generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check completeness
        if 'completeness' in self.validation_results:
            for table, results in self.validation_results['completeness'].items():
                if results['total_rows'] != results['complete_rows']:
                    recommendations.append(f"Address missing values in {table}")

        # Check accuracy
        if 'accuracy' in self.validation_results:
            if 'journal_balance' in self.validation_results['accuracy']:
                if self.validation_results['accuracy']['journal_balance']['unbalanced_count'] > 0:
                    recommendations.append("Review unbalanced journal entries")

        return recommendations