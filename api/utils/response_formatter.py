# api/utils/response_formatter.py

from flask import jsonify
from datetime import datetime
import numpy as np

def format_response(data=None, error=None):
    """Format API response"""
    response = {
        'timestamp': datetime.now().isoformat(),
        'status': 'error' if error else 'success'
    }
    
    if error:
        response['error'] = str(error)
    else:
        response['data'] = convert_to_serializable(data)
    
    return jsonify(response)

def convert_to_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj