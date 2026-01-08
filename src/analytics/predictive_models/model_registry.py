# src/analytics/predictive_models/model_registry.py

import os
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime
import shutil

class ModelRegistry:
    """
    Registry for managing saved models and their metadata
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize registry with base directory
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        self.registry_file = os.path.join(base_dir, 'registry.json')
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize or load registry
        self._initialize_registry()
        
    def _initialize_registry(self) -> None:
        """Initialize or load registry file"""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    self.registry = json.load(f)
            else:
                self.registry = {
                    'models': {},
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._save_registry()
                
        except Exception as e:
            self.logger.error(f"Error initializing registry: {str(e)}")
            raise
            
    def _save_registry(self) -> None:
        """Save registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving registry: {str(e)}")
            raise
            
    def register_model(self,
                      model_type: str,
                      model_version: str,
                      model_path: str,
                      metadata: Optional[Dict] = None) -> str:
        """
        Register a new model
        
        Args:
            model_type: Type of model
            model_version: Version of model
            model_path: Path to model files
            metadata: Optional metadata about the model
            
        Returns:
            Model identifier
        """
        try:
            # Create model identifier
            model_id = f"{model_type}-{model_version}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create model directory
            model_dir = os.path.join(self.base_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy model files
            shutil.copytree(model_path, model_dir, dirs_exist_ok=True)
            
            # Update registry
            self.registry['models'][model_id] = {
                'model_type': model_type,
                'version': model_version,
                'path': model_dir,
                'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': metadata or {}
            }
            
            # Save registry
            self._save_registry()
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            raise
            
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get information about a registered model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dict containing model information or None if not found
        """
        return self.registry['models'].get(model_id)
        
    def get_latest_model(self, model_type: str) -> Optional[Dict]:
        """
        Get latest version of a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Dict containing model information or None if not found
        """
        try:
            # Filter models by type
            type_models = [
                (model_id, info)
                for model_id, info in self.registry['models'].items()
                if info['model_type'] == model_type
            ]
            
            if not type_models:
                return None
                
            # Sort by registration date and get latest
            latest_model = sorted(
                type_models,
                key=lambda x: x[1]['registered_at'],
                reverse=True
            )[0]
            
            return {
                'model_id': latest_model[0],
                **latest_model[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting latest model: {str(e)}")
            raise
            
    def list_models(self,
                   model_type: Optional[str] = None,
                   version: Optional[str] = None) -> List[Dict]:
        """
        List registered models with optional filtering
        
        Args:
            model_type: Optional model type filter
            version: Optional version filter
            
        Returns:
            List of matching model information
        """
        try:
            models = []
            
            for model_id, info in self.registry['models'].items():
                if model_type and info['model_type'] != model_type:
                    continue
                    
                if version and info['version'] != version:
                    continue
                    
                models.append({
                    'model_id': model_id,
                    **info
                })
                
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            raise
            
    def delete_model(self, model_id: str) -> None:
        """
        Delete a registered model
        
        Args:
            model_id: Model identifier to delete
        """
        try:
            if model_id not in self.registry['models']:
                raise ValueError(f"Model {model_id} not found in registry")
                
            # Get model path
            model_path = self.registry['models'][model_id]['path']
            
            # Remove model files
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                
            # Remove from registry
            del self.registry['models'][model_id]
            
            # Save registry
            self._save_registry()
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {str(e)}")
            raise
            
    def update_metadata(self, model_id: str, metadata: Dict) -> None:
        """
        Update metadata for a registered model
        
        Args:
            model_id: Model identifier
            metadata: New metadata to update
        """
        try:
            if model_id not in self.registry['models']:
                raise ValueError(f"Model {model_id} not found in registry")
                
            # Update metadata
            self.registry['models'][model_id]['metadata'].update(metadata)
            
            # Save registry
            self._save_registry()
            
        except Exception as e:
            self.logger.error(f"Error updating metadata: {str(e)}")
            raise