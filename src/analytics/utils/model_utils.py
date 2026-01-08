# src/analytics/utils/model_utils.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class ModelUtils:
    """
    Utility functions for analytics models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_percentage_change(self, 
                                   current_value: float, 
                                   previous_value: float) -> float:
        """
        Calculate percentage change between two values
        """
        try:
            if previous_value == 0:
                raise ValueError("Previous value cannot be zero")
                
            return ((current_value - previous_value) / previous_value) * 100
        
        except Exception as e:
            self.logger.error(f"Error calculating percentage change: {str(e)}")
            raise
            
    def calculate_cagr(self, 
                      start_value: float, 
                      end_value: float, 
                      num_periods: int) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR)
        """
        try:
            if start_value <= 0:
                raise ValueError("Start value must be greater than zero")
                
            cagr = (end_value / start_value) ** (1 / num_periods) - 1
            return cagr * 100
        
        except Exception as e:
            self.logger.error(f"Error calculating CAGR: {str(e)}")
            raise
            
    def calculate_average_growth_rate(self, values: List[float]) -> float:
        """
        Calculate average growth rate over a series of values
        """
        try:
            growth_rates = []
            
            for i in range(1, len(values)):
                growth_rate = self.calculate_percentage_change(
                    values[i], values[i-1]
                )
                growth_rates.append(growth_rate)
                
            return np.mean(growth_rates)
        
        except Exception as e:
            self.logger.error(f"Error calculating average growth rate: {str(e)}")
            raise
            
    def calculate_volatility(self, values: List[float]) -> float:
        """
        Calculate volatility (standard deviation) of a series of values
        """
        try:
            return np.std(values)
        
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            raise
            
    def detect_outliers(self, 
                       values: List[float],
                       threshold: float = 3.0) -> List[Tuple[int, float]]:
        """
        Detect outliers in a series of values using z-score method
        """
        try:
            z_scores = (values - np.mean(values)) / np.std(values)
            outliers = [
                (i, value) 
                for i, (value, z_score) in enumerate(zip(values, z_scores))
                if abs(z_score) > threshold
            ]
            return outliers
        
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            raise
            
    def calculate_moving_average(self, 
                                values: List[float], 
                                window: int) -> List[float]:
        """
        Calculate moving average of a series of values
        """
        try:
            return list(pd.Series(values).rolling(window=window).mean())
        
        except Exception as e:
            self.logger.error(f"Error calculating moving average: {str(e)}")
            raise