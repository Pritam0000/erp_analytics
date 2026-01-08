# config/feature_config.py
"""Configuration for feature engineering"""

# Feature columns from raw data
RAW_FEATURES = [
    'invoice_id',
    'invoice_number',
    'supplier_id',
    'supplier_name',
    'invoice_date',
    'due_date',
    'invoice_amount',
    'tax_amount',
    'po_number',
    'po_amount',
    'invoice_status',
    'invoice_category',
    'payment_terms_days',
    'payment_date',
    'created_date',
    'line_items_count'
]

# Categorical features to encode
CATEGORICAL_FEATURES = [
    'invoice_category',
    'invoice_status',
    'supplier_name'
]

# Numerical features to normalize
NUMERICAL_FEATURES = [
    'invoice_amount',
    'tax_amount',
    'po_amount',
    'payment_terms_days',
    'line_items_count'
]

# Engineered features list
ENGINEERED_FEATURES = [
    'days_until_due',
    'payment_delay_days',
    'invoice_age_days',
    'day_of_week',
    'month',
    'quarter',
    'po_invoice_variance',
    'po_invoice_variance_abs',
    'tax_percentage',
    'amount_per_line_item',
    'supplier_avg_amount',
    'supplier_rejection_rate',
    'supplier_avg_delay',
    'supplier_invoice_count',
    'is_high_value',
    'is_rush',
    'has_po',
    'po_mismatch_flag'
]

# Target variables
TARGET_AMOUNT = 'invoice_amount'
TARGET_DELAY = 'payment_delay_days'
TARGET_REJECTION = 'is_rejected'

# Duplicate detection features
DUPLICATE_FEATURES = [
    'invoice_number',
    'supplier_id',
    'invoice_amount',
    'invoice_date'
]

# Anomaly detection thresholds
ANOMALY_THRESHOLDS = {
    'po_mismatch_percent': 0.10,      # 10% variance
    'high_value_percentile': 0.95,     # 95th percentile
    'rush_days': 7,                    # Less than 7 days
    'duplicate_similarity': 0.80        # 80% similarity
}
