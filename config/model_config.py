# config/model_config.py
"""Configuration for ML models"""

# Random Forest Configuration
RF_REGRESSOR_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

RF_CLASSIFIER_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# XGBoost Configuration
XGB_REGRESSOR_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

XGB_CLASSIFIER_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'scale_pos_weight': 4  # For imbalanced classes
}

# Isolation Forest Configuration
ISOLATION_FOREST_CONFIG = {
    'contamination': 0.05,
    'n_estimators': 100,
    'max_samples': 256,
    'random_state': 42,
    'n_jobs': -1
}

# Local Outlier Factor Configuration
LOF_CONFIG = {
    'n_neighbors': 20,
    'contamination': 0.05,
    'n_jobs': -1
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'categorical_encoding': 'onehot',  # 'onehot' or 'label'
    'numerical_scaling': 'standard',   # 'standard' or 'minmax'
    'handle_missing': 'median',        # 'mean', 'median', or 'drop'
    'outlier_threshold': 3.0           # Z-score threshold
}

# Model Evaluation
TRAIN_TEST_SPLIT = 0.2
CROSS_VALIDATION_FOLDS = 5
RANDOM_STATE = 42
