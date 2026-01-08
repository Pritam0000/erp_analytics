# api/routes/__init__.py

from flask import jsonify
from datetime import datetime
from .financial_routes import register_financial_routes

def register_routes(app):
    """Register all application routes"""
    
    @app.route('/')
    def home():
        """Home page"""
        return jsonify({
            'status': 'ok',
            'version': '1.0.0',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    @app.route('/api/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'database': 'mock' if app.config.get('USE_MOCK_DB', True) else 'oracle',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    # Register financial routes
    register_financial_routes(app)