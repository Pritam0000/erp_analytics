# api/routes/financial_routes.py

from flask import Blueprint, jsonify, request, current_app
from datetime import datetime, timedelta

def register_financial_routes(app):
    """Register financial routes with the Flask app"""

    @app.route('/api/accounts')
    def get_accounts():
        """Get chart of accounts with optional filters"""
        try:
            account_type = request.args.get('type')
            is_active = request.args.get('active', 'Y')
            
            query = """
                SELECT 
                    account_id,
                    account_number,
                    account_name,
                    account_type,
                    account_category,
                    is_active
                FROM GL_COA
                WHERE 1=1
            """
            
            params = {}
            if account_type:
                query += " AND account_type = :account_type"
                params['account_type'] = account_type
                
            if is_active:
                query += " AND is_active = :is_active"
                params['is_active'] = is_active
                
            query += " ORDER BY account_number"
            
            df = app.db.execute_query(query, params)
            return jsonify(df.to_dict(orient='records'))
            
        except Exception as e:
            app.logger.error(f"Error fetching accounts: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/journals')
    def get_journals():
        """Get journal entries with optional date range"""
        try:
            start_date = request.args.get('start_date', 
                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            end_date = request.args.get('end_date', 
                datetime.now().strftime('%Y-%m-%d'))
            status = request.args.get('status', 'POSTED')
            
            query = """
                SELECT 
                    j.journal_id,
                    j.journal_number,
                    j.journal_date,
                    j.status,
                    COUNT(l.line_id) as line_count,
                    SUM(l.debit_amount) as total_debits,
                    SUM(l.credit_amount) as total_credits
                FROM GL_JOURNALS j
                LEFT JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                WHERE j.journal_date BETWEEN :start_date AND :end_date
            """
            
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            
            if status:
                query += " AND j.status = :status"
                params['status'] = status
                
            query += """
                GROUP BY 
                    j.journal_id,
                    j.journal_number,
                    j.journal_date,
                    j.status
                ORDER BY j.journal_date DESC
            """
            
            df = app.db.execute_query(query, params)
            return jsonify(df.to_dict(orient='records'))
            
        except Exception as e:
            app.logger.error(f"Error fetching journals: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/journal/<int:journal_id>')
    def get_journal_details(journal_id):
        """Get detailed information for a specific journal entry"""
        try:
            # Get journal header
            header_query = """
                SELECT 
                    journal_id,
                    journal_number,
                    journal_date,
                    status,
                    created_date,
                    created_by
                FROM GL_JOURNALS
                WHERE journal_id = :journal_id
            """
            
            header_df = app.db.execute_query(header_query, {'journal_id': journal_id})
            if header_df.empty:
                return jsonify({'error': 'Journal not found'}), 404
                
            # Get journal lines
            lines_query = """
                SELECT 
                    l.line_id,
                    l.account_id,
                    c.account_number,
                    c.account_name,
                    l.debit_amount,
                    l.credit_amount,
                    l.description
                FROM GL_JOURNAL_LINES l
                JOIN GL_COA c ON l.account_id = c.account_id
                WHERE l.journal_id = :journal_id
                ORDER BY l.line_id
            """
            
            lines_df = app.db.execute_query(lines_query, {'journal_id': journal_id})
            
            return jsonify({
                'header': header_df.iloc[0].to_dict(),
                'lines': lines_df.to_dict(orient='records')
            })
            
        except Exception as e:
            app.logger.error(f"Error fetching journal details: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/accounts/balances')
    def get_account_balances():
        """Get account balances as of a specific date"""
        try:
            as_of_date = request.args.get('as_of_date', 
                datetime.now().strftime('%Y-%m-%d'))
                
            query = """
                WITH account_activity AS (
                    SELECT 
                        l.account_id,
                        SUM(l.debit_amount) as total_debits,
                        SUM(l.credit_amount) as total_credits
                    FROM GL_JOURNAL_LINES l
                    JOIN GL_JOURNALS j ON l.journal_id = j.journal_id
                    WHERE j.journal_date <= :as_of_date
                    AND j.status = 'POSTED'
                    GROUP BY l.account_id
                )
                SELECT 
                    c.account_id,
                    c.account_number,
                    c.account_name,
                    c.account_type,
                    c.account_category,
                    COALESCE(a.total_debits, 0) as total_debits,
                    COALESCE(a.total_credits, 0) as total_credits,
                    COALESCE(a.total_debits - a.total_credits, 0) as balance
                FROM GL_COA c
                LEFT JOIN account_activity a ON c.account_id = a.account_id
                WHERE c.is_active = 'Y'
                ORDER BY c.account_number
            """
            
            df = app.db.execute_query(query, {'as_of_date': as_of_date})
            return jsonify(df.to_dict(orient='records'))
            
        except Exception as e:
            app.logger.error(f"Error fetching account balances: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/accounts/<int:account_id>/activity')
    def get_account_activity(account_id):
        """Get activity for a specific account"""
        try:
            start_date = request.args.get('start_date', 
                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            end_date = request.args.get('end_date', 
                datetime.now().strftime('%Y-%m-%d'))
            
            query = """
                SELECT 
                    j.journal_date,
                    j.journal_number,
                    l.debit_amount,
                    l.credit_amount,
                    l.description,
                    j.status
                FROM GL_JOURNAL_LINES l
                JOIN GL_JOURNALS j ON l.journal_id = j.journal_id
                WHERE l.account_id = :account_id
                AND j.journal_date BETWEEN :start_date AND :end_date
                ORDER BY j.journal_date, j.journal_id
            """
            
            df = app.db.execute_query(query, {
                'account_id': account_id,
                'start_date': start_date,
                'end_date': end_date
            })
            
            return jsonify(df.to_dict(orient='records'))
            
        except Exception as e:
            app.logger.error(f"Error fetching account activity: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # Additional helper route for summary data
    @app.route('/api/summary')
    def get_financial_summary():
        """Get high-level financial summary"""
        try:
            as_of_date = request.args.get('as_of_date', 
                datetime.now().strftime('%Y-%m-%d'))
            
            # Get account balances by type
            query = """
                WITH account_balances AS (
                    SELECT 
                        c.account_type,
                        SUM(COALESCE(l.debit_amount, 0) - COALESCE(l.credit_amount, 0)) as balance
                    FROM GL_COA c
                    LEFT JOIN GL_JOURNAL_LINES l ON c.account_id = l.account_id
                    LEFT JOIN GL_JOURNALS j ON l.journal_id = j.journal_id
                    WHERE j.journal_date <= :as_of_date
                    AND j.status = 'POSTED'
                    GROUP BY c.account_type
                )
                SELECT 
                    account_type,
                    balance
                FROM account_balances
                ORDER BY account_type
            """
            
            df = app.db.execute_query(query, {'as_of_date': as_of_date})
            
            summary = {
                'as_of_date': as_of_date,
                'balances': df.to_dict(orient='records'),
                'total_accounts': len(df),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return jsonify(summary)
            
        except Exception as e:
            app.logger.error(f"Error fetching financial summary: {str(e)}")
            return jsonify({'error': str(e)}), 500