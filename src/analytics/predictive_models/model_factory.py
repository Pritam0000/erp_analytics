# src/analytics/predictive_models/model_factory.py

from typing import Dict, Optional, Type, Union
import logging
from datetime import datetime

from src.data_loading.data_loader import FinancialDataLoader
from . import AVAILABLE_MODELS, DEFAULT_PARAMS

class ModelFactory:
    """
    Factory class for creating and managing predictive models
    """
    
    def __init__(self, data_loader: FinancialDataLoader):
        """
        Initialize factory with data loader
        
        Args:
            data_loader: Instance of FinancialDataLoader
        """
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        self._model_instances = {}
        
    def get_model(self, 
                 model_type: str,
                 load_path: Optional[str] = None) -> Union[Type, None]:
        """
        Get or create a model instance
        
        Args:
            model_type: Type of model to create ('cash_flow', 'payment', etc.)
            load_path: Optional path to load saved model from
            
        Returns:
            Model instance or None if type not found
        """
        try:
            if model_type not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model type: {model_type}")
                
            # Check if model instance exists
            if model_type not in self._model_instances:
                # Create new instance
                model_class = AVAILABLE_MODELS[model_type]
                self._model_instances[model_type] = model_class(self.data_loader)
                
                # Load saved model if path provided
                if load_path:
                    self._model_instances[model_type].load_model(load_path)
                    
            return self._model_instances[model_type]
            
        except Exception as e:
            self.logger.error(f"Error getting model instance: {str(e)}")
            raise
            
    def train_model(self,
                   model_type: str,
                   start_date: datetime,
                   end_date: datetime,
                   params: Optional[Dict] = None,
                   save_path: Optional[str] = None) -> Dict:
        """
        Train a specific model type
        
        Args:
            model_type: Type of model to train
            start_date: Start of training period
            end_date: End of training period
            params: Optional training parameters
            save_path: Optional path to save trained model
            
        Returns:
            Dict containing training results
        """
        try:
            # Get model instance
            model = self.get_model(model_type)
            
            # Prepare training parameters
            train_params = DEFAULT_PARAMS.copy()
            if params:
                train_params.update(params)
                
            # Prepare training data
            training_data = model.prepare_training_data(start_date, end_date)
            
            # Train model
            results = model.train_model(training_data, **train_params)
            
            # Save model if path provided
            if save_path:
                model.save_model(save_path)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def get_model_summary(self, model_type: str) -> Dict:
        """
        Get summary of trained model
        
        Args:
            model_type: Type of model to summarize
            
        Returns:
            Dict containing model summary
        """
        try:
            model = self.get_model(model_type)
            
            summary = {
                'model_type': model_type,
                'training_period': getattr(model, 'training_period', None),
                'metrics': getattr(model, 'model_metrics', {}),
                'feature_count': len(getattr(model, 'feature_columns', [])),
                'is_trained': model.model is not None
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {str(e)}")
            raise
            
    def delete_model(self, model_type: str) -> None:
        """
        Delete model instance from factory
        
        Args:
            model_type: Type of model to delete
        """
        try:
            if model_type in self._model_instances:
                del self._model_instances[model_type]
                
        except Exception as e:
            self.logger.error(f"Error deleting model: {str(e)}")
            raise