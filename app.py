# app.py

import os
from flask import Flask
from src.utils.database_utils import DatabaseConnector
from api.routes import register_routes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Load default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        USE_MOCK_DB=os.getenv('USE_MOCK_DB', 'true').lower() == 'true'
    )

    # Load additional configuration if provided
    if config is not None:
        app.config.update(config)

    # Initialize database connection
    app.db = DatabaseConnector()

    # Register routes
    register_routes(app)

    # Add error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Not Found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal Server Error'}, 500

    return app


# Create the application instance
app = create_app()