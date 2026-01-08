# api/__init__.py
from flask import Flask
from api.routes import register_routes

def create_app(config=None):
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        USE_MOCK_DB=True
    )
    
    # Override with any provided config
    if config:
        app.config.update(config)
        
    # Register all routes
    register_routes(app)
    
    return app