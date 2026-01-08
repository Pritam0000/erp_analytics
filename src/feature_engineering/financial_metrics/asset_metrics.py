# src/feature_engineering/financial_metrics/asset_metrics.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Tuple

from src.utils.database_utils import DatabaseConnector

class AssetMetricsCalculator:
    """
    Class to calculate and analyze fixed asset metrics including:
    - Depreciation analysis
    - Asset utilization
    - Category performance
    - Asset lifecycle metrics
    """
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger for asset metrics"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # If logger already has handlers, don't add more
        if not logger.handlers:
            # Create file handler using existing logs directory
            log_file = os.path.join('logs', 'feature_engineering', 'asset_metrics.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            # Create formatter matching existing logging format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            # Add handler
            logger.addHandler(handler)
        
        return logger

    def calculate_asset_depreciation(self, asset_id: int, 
                                   as_of_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate depreciation metrics for a specific asset
        """
        self.logger.info(f"Calculating depreciation for asset {asset_id}")
        try:
            as_of_date = as_of_date or datetime.now()
            
            query = """
                SELECT 
                    a.asset_id,
                    a.acquisition_date,
                    a.acquisition_cost,
                    a.salvage_value,
                    a.life_years,
                    c.depreciation_method,
                    d.accumulated_depreciation,
                    d.book_value
                FROM FA_ASSETS a
                JOIN FA_CATEGORIES c ON a.category_id = c.category_id
                LEFT JOIN FA_DEPRECIATION d ON a.asset_id = d.asset_id
                WHERE a.asset_id = :asset_id
                AND (d.period_date IS NULL OR d.period_date = (
                    SELECT MAX(period_date) 
                    FROM FA_DEPRECIATION 
                    WHERE asset_id = a.asset_id
                    AND period_date <= :as_of_date
                ))
            """
            
            df = self.db.execute_query(
                query,
                params={'asset_id': asset_id, 'as_of_date': as_of_date}
            )
            
            if df.empty:
                raise ValueError(f"Asset {asset_id} not found")
                
            row = df.iloc[0]
            
            # Calculate annual depreciation rate
            annual_depreciation = self._calculate_annual_depreciation(
                acquisition_cost=float(row['acquisition_cost']),
                salvage_value=float(row['salvage_value']),
                life_years=int(row['life_years']),
                depreciation_method=str(row['depreciation_method'])
            )
            
            # Calculate current values
            years_held = (as_of_date - row['acquisition_date']).days / 365.25
            depreciation_factor = min(years_held / row['life_years'], 1.0)
            
            expected_accumulated_depreciation = annual_depreciation * years_held
            expected_book_value = (row['acquisition_cost'] - expected_accumulated_depreciation)
            
            current_book_value = row['book_value'] if pd.notnull(row['book_value']) else expected_book_value
            accumulated_depreciation = (row['accumulated_depreciation'] 
                                     if pd.notnull(row['accumulated_depreciation']) 
                                     else expected_accumulated_depreciation)
            
            return {
                'acquisition_cost': float(row['acquisition_cost']),
                'current_book_value': float(current_book_value),
                'accumulated_depreciation': float(accumulated_depreciation),
                'annual_depreciation_rate': (annual_depreciation / float(row['acquisition_cost'])) * 100,
                'remaining_life_years': max(float(row['life_years']) - years_held, 0),
                'depreciation_percentage': depreciation_factor * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating depreciation for asset {asset_id}: {str(e)}")
            raise

    def _calculate_annual_depreciation(self, acquisition_cost: float, 
                                     salvage_value: float,
                                     life_years: int,
                                     depreciation_method: str) -> float:
        """Calculate annual depreciation based on method"""
        depreciable_amount = acquisition_cost - salvage_value
        
        if depreciation_method == 'STRAIGHT_LINE':
            return depreciable_amount / life_years
        elif depreciation_method == 'DECLINING_BALANCE':
            # Double declining balance
            rate = 2.0 / life_years
            return acquisition_cost * rate
        elif depreciation_method == 'SUM_OF_YEARS':
            # Sum of years digits
            sum_of_years = (life_years * (life_years + 1)) / 2
            first_year_fraction = life_years / sum_of_years
            return depreciable_amount * first_year_fraction
        else:
            raise ValueError(f"Unsupported depreciation method: {depreciation_method}")

    def analyze_category_performance(self, category_id: Optional[int] = None,
                                   as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Analyze asset performance by category
        """
        self.logger.info(f"Analyzing category performance for category {category_id if category_id else 'all'}")
        try:
            as_of_date = as_of_date or datetime.now()
            
            query = """
                WITH AssetMetrics AS (
                    SELECT 
                        c.category_id,
                        c.category_name,
                        c.depreciation_method,
                        COUNT(a.asset_id) as asset_count,
                        SUM(a.acquisition_cost) as total_acquisition_cost,
                        SUM(a.salvage_value) as total_salvage_value,
                        AVG(a.life_years) as avg_life_years,
                        SUM(CASE WHEN a.status = 'IN_SERVICE' THEN 1 ELSE 0 END) as active_assets,
                        SUM(CASE WHEN a.status = 'RETIRED' THEN 1 ELSE 0 END) as retired_assets,
                        SUM(CASE WHEN a.status = 'DISPOSED' THEN 1 ELSE 0 END) as disposed_assets
                    FROM FA_CATEGORIES c
                    LEFT JOIN FA_ASSETS a ON c.category_id = a.category_id
                    WHERE (:category_id IS NULL OR c.category_id = :category_id)
                    GROUP BY c.category_id, c.category_name, c.depreciation_method
                )
                SELECT 
                    am.*,
                    ROUND(am.active_assets * 100.0 / NULLIF(am.asset_count, 0), 2) as utilization_rate
                FROM AssetMetrics am
            """
            
            df = self.db.execute_query(
                query,
                params={'category_id': category_id, 'as_of_date': as_of_date}
            )
            
            if df.empty:
                return pd.DataFrame()
                
            # Calculate additional metrics
            df['avg_asset_cost'] = df['total_acquisition_cost'] / df['asset_count']
            df['value_retention'] = ((df['total_acquisition_cost'] - df['total_salvage_value']) / 
                                   df['total_acquisition_cost']) * 100
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error analyzing category performance: {str(e)}")
            raise

    def get_asset_lifecycle_metrics(self, asset_id: int) -> Dict[str, any]:
        """
        Get comprehensive lifecycle metrics for an asset
        """
        self.logger.info(f"Getting lifecycle metrics for asset {asset_id}")
        try:
            # Get asset details
            query = """
                SELECT 
                    a.asset_id,
                    a.asset_number,
                    a.asset_name,
                    a.category_id,
                    c.category_name,
                    a.acquisition_date,
                    a.acquisition_cost,
                    a.salvage_value,
                    a.life_years,
                    a.status,
                    c.depreciation_method
                FROM FA_ASSETS a
                JOIN FA_CATEGORIES c ON a.category_id = c.category_id
                WHERE a.asset_id = :asset_id
            """
            
            df = self.db.execute_query(query, params={'asset_id': asset_id})
            
            if df.empty:
                raise ValueError(f"Asset {asset_id} not found")
                
            asset_info = df.iloc[0].to_dict()
            
            # Calculate current metrics
            depreciation_metrics = self.calculate_asset_depreciation(asset_id)
            
            # Get depreciation history
            depreciation_query = """
                SELECT 
                    period_date,
                    depreciation_amount,
                    accumulated_depreciation,
                    book_value
                FROM FA_DEPRECIATION
                WHERE asset_id = :asset_id
                ORDER BY period_date
            """
            
            depreciation_df = self.db.execute_query(
                depreciation_query,
                params={'asset_id': asset_id}
            )
            
            # Calculate lifecycle metrics
            acquisition_date = asset_info['acquisition_date']
            years_held = (datetime.now() - acquisition_date).days / 365.25
            
            lifecycle_stage = self._determine_lifecycle_stage(
                years_held,
                asset_info['life_years'],
                asset_info['status']
            )
            
            return {
                'asset_info': asset_info,
                'lifecycle_metrics': {
                    'age_years': years_held,
                    'percent_through_lifecycle': min(years_held / asset_info['life_years'] * 100, 100),
                    'lifecycle_stage': lifecycle_stage,
                    'years_remaining': max(asset_info['life_years'] - years_held, 0)
                },
                'financial_metrics': {
                    'total_depreciation': depreciation_df['depreciation_amount'].sum() if not depreciation_df.empty else 0,
                    'current_book_value': depreciation_metrics['current_book_value'],
                    'value_retention': (depreciation_metrics['current_book_value'] / 
                                      asset_info['acquisition_cost'] * 100),
                    'depreciation_rate': depreciation_metrics['annual_depreciation_rate']
                },
                'depreciation_history': depreciation_df.to_dict('records') if not depreciation_df.empty else [],
                'performance_indicators': {
                    'utilization_status': self._determine_utilization_status(asset_info['status']),
                    'depreciation_variance': self._calculate_depreciation_variance(
                        depreciation_df, 
                        asset_info['acquisition_cost'],
                        asset_info['life_years']
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting lifecycle metrics for asset {asset_id}: {str(e)}")
            raise

    def _determine_lifecycle_stage(self, years_held: float, 
                                 life_years: int,
                                 status: str) -> str:
        """Determine asset lifecycle stage"""
        if status != 'IN_SERVICE':
            return status
            
        lifecycle_percent = (years_held / life_years) * 100
        
        if lifecycle_percent < 20:
            return 'EARLY_LIFE'
        elif lifecycle_percent < 40:
            return 'MID_LIFE'
        elif lifecycle_percent < 60:
            return 'MATURE'
        elif lifecycle_percent < 80:
            return 'LATE_LIFE'
        else:
            return 'END_OF_LIFE'

    def _determine_utilization_status(self, status: str) -> str:
        """Determine asset utilization status"""
        status_map = {
            'IN_SERVICE': 'ACTIVE',
            'RETIRED': 'INACTIVE',
            'DISPOSED': 'TERMINATED'
        }
        return status_map.get(status, 'UNKNOWN')

    def _calculate_depreciation_variance(self, depreciation_df: pd.DataFrame,
                                       acquisition_cost: float,
                                       life_years: int) -> float:
        """Calculate variance from expected depreciation"""
        if depreciation_df.empty:
            return 0.0
            
        # Expected annual depreciation (simple straight-line for variance)
        expected_annual = acquisition_cost / life_years
        
        # Calculate average actual annual depreciation
        actual_annual = depreciation_df['depreciation_amount'].mean() * 12  # Assuming monthly periods
        
        # Calculate variance percentage
        if expected_annual == 0:
            return 0.0
            
        return ((actual_annual - expected_annual) / expected_annual) * 100