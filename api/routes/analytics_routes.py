# api/routes/analytics_routes.py

from flask import Blueprint, request, jsonify
from datetime import datetime
from api.services.analytics_service import AnalyticsService
from api.utils.response_formatter import format_response

bp = Blueprint('analytics', __name__)
service = AnalyticsService()

@bp.route('/balance-sheet', methods=['GET'])
def get_balance_sheet():
    """Get balance sheet analysis"""
    try:
        as_of_date = request.args.get('as_of_date')
        if as_of_date:
            as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d')
            
        result = service.get_balance_sheet_analysis(as_of_date)
        return format_response(result)
    except Exception as e:
        return format_response(error=str(e)), 500

@bp.route('/income-statement', methods=['GET'])
def get_income_statement():
    """Get income statement analysis"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        result = service.get_income_statement_analysis(start_date, end_date)
        return format_response(result)
    except Exception as e:
        return format_response(error=str(e)), 500

@bp.route('/cash-flow', methods=['GET'])
def get_cash_flow():
    """Get cash flow analysis"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        result = service.get_cash_flow_analysis(start_date, end_date)
        return format_response(result)
    except Exception as e:
        return format_response(error=str(e)), 500

@bp.route('/ratios', methods=['GET'])
def get_financial_ratios():
    """Get financial ratios analysis"""
    try:
        as_of_date = request.args.get('as_of_date')
        if as_of_date:
            as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d')
            
        result = service.get_financial_ratios(as_of_date)
        return format_response(result)
    except Exception as e:
        return format_response(error=str(e)), 500