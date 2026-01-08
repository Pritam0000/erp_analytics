# src/feature_engineering/risk_indicators/risk_indicators.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats

from src.utils.database_utils import DatabaseConnector

class RiskIndicatorCalculator:
    """
    Class to calculate and analyze financial risk indicators including:
    - Payment default probability
    - Account balance volatility
    - Transaction anomaly scores
    - Operational risk metrics
    - Exposure analytics
    """
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger for risk analysis"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = os.path.join('logs', 'feature_engineering', 'risk_indicators.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
            def cleanup():
                for handler in logger.handlers:
                    handler.close()
            import atexit
            atexit.register(cleanup)
            
        return logger
    
    def analyze_payment_default_risk(self, 
                                supplier_id: Optional[int] = None,
                                lookback_days: int = 365) -> Dict[str, any]:
        """
        Analyze payment default risk based on historical payment behavior
        
        Args:
            supplier_id: Optional supplier ID to analyze
            lookback_days: Number of days of historical data to analyze
            
        Returns:
            Dict containing payment default risk metrics
        """
        try:
            start_date = datetime.now() - timedelta(days=lookback_days)
            
            query = """
                WITH PaymentHistory AS (
                    SELECT 
                        i.supplier_id,
                        i.invoice_id,
                        i.invoice_date,
                        i.due_date,
                        i.amount as invoice_amount,
                        COALESCE(p.payment_date, CURRENT_DATE) as payment_date,
                        COALESCE(p.amount, 0) as paid_amount,
                        CASE 
                            WHEN p.amount IS NULL OR p.amount < i.amount THEN 1 
                            ELSE 0 
                        END as is_defaulted,
                        EXTRACT(DAY FROM CURRENT_DATE - i.due_date) as days_outstanding
                    FROM AP_INVOICES i
                    LEFT JOIN AP_PAYMENTS p ON i.invoice_id = p.invoice_id
                    WHERE i.invoice_date >= :start_date
                        {supplier_filter}
                )
                SELECT 
                    supplier_id,
                    COUNT(*) as total_invoices,
                    COUNT(CASE WHEN is_defaulted = 1 THEN 1 END) as defaulted_invoices,
                    AVG(days_outstanding) as avg_days_outstanding,
                    MAX(days_outstanding) as max_days_outstanding,
                    SUM(invoice_amount) as total_exposure,
                    SUM(paid_amount) as total_paid,
                    COUNT(CASE WHEN days_outstanding > 30 THEN 1 END) as overdue_30_plus,
                    COUNT(CASE WHEN days_outstanding > 90 THEN 1 END) as overdue_90_plus
                FROM PaymentHistory
                GROUP BY supplier_id
            """
            
            params = {'start_date': start_date}
            supplier_filter = ""
            
            if supplier_id:
                supplier_filter = "AND i.supplier_id = :supplier_id"
                params['supplier_id'] = supplier_id
                
            query = query.format(supplier_filter=supplier_filter)
            
            df = self.db.execute_query(query, params=params)
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'period': {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': datetime.now().strftime('%Y-%m-%d')
                    }
                }
                
            risk_metrics = []
            for _, row in df.iterrows():
                # Calculate risk components
                default_rate = row['defaulted_invoices'] / row['total_invoices']
                payment_delay = row['avg_days_outstanding']
                exposure_ratio = 1 - (row['total_paid'] / row['total_exposure'])
                late_payment_ratio = row['overdue_30_plus'] / row['total_invoices']
                
                # Calculate composite risk score
                risk_score = self._calculate_payment_risk_score(
                    default_rate=default_rate,
                    payment_delay=payment_delay,
                    exposure_ratio=exposure_ratio,
                    late_payment_ratio=late_payment_ratio
                )
                
                risk_metrics.append({
                    'supplier_id': row['supplier_id'],
                    'risk_score': risk_score,
                    'risk_level': self._determine_risk_level(risk_score),
                    'metrics': {
                        'default_rate': round(default_rate * 100, 2),
                        'avg_days_outstanding': round(payment_delay, 1),
                        'exposure_ratio': round(exposure_ratio * 100, 2),
                        'late_payment_ratio': round(late_payment_ratio * 100, 2),
                        'total_exposure': float(row['total_exposure']),
                        'total_invoices': int(row['total_invoices']),
                        'overdue_30_plus': int(row['overdue_30_plus']),
                        'overdue_90_plus': int(row['overdue_90_plus'])
                    }
                })
            
            return {
                'status': 'ANALYZED',
                'risk_metrics': risk_metrics,
                'period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'lookback_days': lookback_days
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing payment default risk: {str(e)}")
            raise
            
    def _calculate_payment_risk_score(self,
                                    default_rate: float,
                                    payment_delay: float,
                                    exposure_ratio: float,
                                    late_payment_ratio: float) -> float:
        """Calculate composite payment risk score"""
        # Normalize payment delay (cap at 180 days)
        normalized_delay = min(payment_delay / 180, 1.0)
        
        # Weight factors
        weights = {
            'default_rate': 0.35,
            'payment_delay': 0.25,
            'exposure_ratio': 0.25,
            'late_payment': 0.15
        }
        
        # Calculate weighted score
        risk_score = (
            weights['default_rate'] * default_rate +
            weights['payment_delay'] * normalized_delay +
            weights['exposure_ratio'] * exposure_ratio +
            weights['late_payment'] * late_payment_ratio
        )
        
        return round(risk_score * 100, 2)
    
    def analyze_account_volatility(self,
                                account_id: int,
                                period_months: int = 12) -> Dict[str, any]:
        """
        Analyze account balance volatility and risk patterns
        
        Args:
            account_id: The GL account ID to analyze
            period_months: Number of months to analyze
            
        Returns:
            Dict containing volatility metrics
        """
        try:
            start_date = datetime.now() - timedelta(days=period_months * 30)
            
            query = """
                WITH DailyBalances AS (
                    SELECT 
                        j.journal_date as transaction_date,
                        SUM(gl.debit_amount - gl.credit_amount) as net_change,
                        SUM(SUM(gl.debit_amount - gl.credit_amount)) 
                            OVER (ORDER BY j.journal_date) as running_balance
                    FROM GL_JOURNAL_LINES gl
                    JOIN GL_JOURNALS j ON gl.journal_id = j.journal_id
                    WHERE gl.account_id = :account_id
                        AND j.journal_date >= :start_date
                        AND j.status = 'POSTED'
                    GROUP BY j.journal_date
                    ORDER BY j.journal_date
                )
                SELECT 
                    transaction_date,
                    net_change,
                    running_balance,
                    running_balance - LAG(running_balance, 1) OVER (ORDER BY transaction_date)
                        as daily_change
                FROM DailyBalances
            """
            
            df = self.db.execute_query(
                query,
                params={
                    'account_id': account_id,
                    'start_date': start_date
                }
            )
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'account_id': account_id,
                    'period_months': period_months
                }
            
            # Calculate volatility metrics
            balance_volatility = df['running_balance'].std()
            daily_volatility = df['daily_change'].std()
            
            # Calculate trend
            df['days_from_start'] = (df['transaction_date'] - df['transaction_date'].min()).dt.days
            trend_coef = np.polyfit(df['days_from_start'], df['running_balance'], 1)[0]
            
            # Detect extreme movements
            daily_changes = df['daily_change'].dropna()
            mean_change = daily_changes.mean()
            std_change = daily_changes.std()
            threshold = 2.0  # Standard deviations for extreme movement
            
            extreme_movements = df[
                abs(df['daily_change'] - mean_change) > threshold * std_change
            ][['transaction_date', 'daily_change']].to_dict('records')
            
            # Calculate stability metrics
            stability_metrics = {
                'volatility_score': self._calculate_volatility_score(
                    balance_volatility=balance_volatility,
                    daily_volatility=daily_volatility,
                    extreme_movements=len(extreme_movements)
                ),
                'trend_direction': 'INCREASING' if trend_coef > 0 else 'DECREASING',
                'trend_strength': abs(trend_coef),
                'extreme_movements': extreme_movements,
                'statistics': {
                    'mean_balance': float(df['running_balance'].mean()),
                    'max_balance': float(df['running_balance'].max()),
                    'min_balance': float(df['running_balance'].min()),
                    'balance_volatility': float(balance_volatility),
                    'daily_volatility': float(daily_volatility),
                    'extreme_movement_count': len(extreme_movements)
                }
            }
            
            # Calculate risk level
            risk_level = self._assess_volatility_risk(
                volatility_score=stability_metrics['volatility_score'],
                extreme_movement_count=len(extreme_movements),
                trend_strength=stability_metrics['trend_strength']
            )
            
            return {
                'status': 'ANALYZED',
                'account_id': account_id,
                'risk_level': risk_level,
                'stability_metrics': stability_metrics,
                'period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'months_analyzed': period_months
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing account volatility: {str(e)}")
            raise
            
    def _calculate_volatility_score(self,
                                  balance_volatility: float,
                                  daily_volatility: float,
                                  extreme_movements: int) -> float:
        """Calculate composite volatility score"""
        # Normalize components
        max_daily_volatility = 10000  # Adjust based on business context
        max_balance_volatility = 100000  # Adjust based on business context
        max_extreme_movements = 10  # Adjust based on business context
        
        normalized_daily = min(daily_volatility / max_daily_volatility, 1.0)
        normalized_balance = min(balance_volatility / max_balance_volatility, 1.0)
        normalized_extremes = min(extreme_movements / max_extreme_movements, 1.0)
        
        # Weight factors
        weights = {
            'daily_volatility': 0.4,
            'balance_volatility': 0.4,
            'extreme_movements': 0.2
        }
        
        # Calculate weighted score
        volatility_score = (
            weights['daily_volatility'] * normalized_daily +
            weights['balance_volatility'] * normalized_balance +
            weights['extreme_movements'] * normalized_extremes
        )
        
        return round(volatility_score * 100, 2)
        
    def _assess_volatility_risk(self,
                               volatility_score: float,
                               extreme_movement_count: int,
                               trend_strength: float) -> str:
        """Assess overall volatility risk level"""
        # Base risk on volatility score
        if volatility_score <= 30:
            risk_level = 'LOW'
        elif volatility_score <= 70:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
            
        # Adjust for extreme movements
        if extreme_movement_count > 5 and risk_level == 'LOW':
            risk_level = 'MEDIUM'
        elif extreme_movement_count > 8:
            risk_level = 'HIGH'
            
        # Adjust for strong trends
        if trend_strength > 0.5 and risk_level != 'HIGH':
            risk_level = 'MEDIUM'
            
        return risk_level
    
    def analyze_transaction_anomalies(self,
                                    lookback_days: int = 180,
                                    threshold_stddev: float = 3.0) -> Dict[str, any]:
        """
        Analyze transactions for anomalous patterns and unusual behavior
        
        Args:
            lookback_days: Number of days of historical data to analyze
            threshold_stddev: Number of standard deviations for anomaly detection
            
        Returns:
            Dict containing anomaly analysis results
        """
        try:
            start_date = datetime.now() - timedelta(days=lookback_days)
            
            # Query transaction patterns
            query = """
                WITH TransactionStats AS (
                    SELECT 
                        j.journal_id,
                        j.journal_date,
                        COUNT(l.line_id) as line_count,
                        COUNT(DISTINCT l.account_id) as account_count,
                        SUM(l.debit_amount) as total_debits,
                        SUM(l.credit_amount) as total_credits,
                        MAX(GREATEST(l.debit_amount, l.credit_amount)) as max_amount
                    FROM GL_JOURNALS j
                    JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                    WHERE j.journal_date >= :start_date
                        AND j.status = 'POSTED'
                    GROUP BY j.journal_id, j.journal_date
                )
                SELECT 
                    journal_id,
                    journal_date,
                    line_count,
                    account_count,
                    total_debits,
                    total_credits,
                    max_amount,
                    AVG(line_count) OVER (
                        ORDER BY journal_date 
                        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
                    ) as avg_line_count,
                    AVG(max_amount) OVER (
                        ORDER BY journal_date 
                        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
                    ) as avg_amount,
                    STDDEV(max_amount) OVER (
                        ORDER BY journal_date 
                        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
                    ) as stddev_amount
                FROM TransactionStats
                ORDER BY journal_date
            """
            
            df = self.db.execute_query(
                query,
                params={'start_date': start_date}
            )
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'period': {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': datetime.now().strftime('%Y-%m-%d')
                    }
                }
            
            # Detect amount-based anomalies
            amount_anomalies = df[
                abs(df['max_amount'] - df['avg_amount']) > 
                threshold_stddev * df['stddev_amount']
            ]
            
            # Detect pattern-based anomalies
            pattern_anomalies = self._detect_pattern_anomalies(
                df, threshold_stddev
            )
            
            # Combine anomalies and calculate risk metrics
            all_anomalies = pd.concat([
                amount_anomalies[['journal_id', 'journal_date', 'max_amount']].assign(type='AMOUNT'),
                pattern_anomalies[['journal_id', 'journal_date', 'max_amount']].assign(type='PATTERN')
            ]).sort_values('journal_date')
            
            anomaly_metrics = {
                'total_transactions': len(df),
                'amount_anomalies': len(amount_anomalies),
                'pattern_anomalies': len(pattern_anomalies),
                'total_anomalies': len(all_anomalies),
                'anomaly_rate': round(len(all_anomalies) / len(df) * 100, 2)
            }
            
            # Calculate time-based clustering of anomalies
            anomaly_clusters = self._analyze_anomaly_clustering(all_anomalies)
            
            # Calculate overall risk metrics
            risk_metrics = self._calculate_anomaly_risk_metrics(
                anomaly_metrics, anomaly_clusters
            )
            
            return {
                'status': 'ANALYZED',
                'anomaly_metrics': anomaly_metrics,
                'risk_metrics': risk_metrics,
                'anomalies': {
                    'amount_based': amount_anomalies[
                        ['journal_id', 'journal_date', 'max_amount', 'avg_amount', 'stddev_amount']
                    ].to_dict('records'),
                    'pattern_based': pattern_anomalies[
                        ['journal_id', 'journal_date', 'line_count', 'account_count']
                    ].to_dict('records')
                },
                'clustering_analysis': anomaly_clusters,
                'period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'lookback_days': lookback_days
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing transaction anomalies: {str(e)}")
            raise
            
    def _detect_pattern_anomalies(self,
                                df: pd.DataFrame,
                                threshold_stddev: float) -> pd.DataFrame:
        """Detect anomalies based on transaction patterns"""
        # Calculate rolling statistics for pattern metrics
        df['avg_line_count'] = df['line_count'].rolling(window=30, min_periods=1).mean()
        df['stddev_line_count'] = df['line_count'].rolling(window=30, min_periods=1).std()
        
        df['avg_account_count'] = df['account_count'].rolling(window=30, min_periods=1).mean()
        df['stddev_account_count'] = df['account_count'].rolling(window=30, min_periods=1).std()
        
        # Detect anomalies based on multiple criteria
        line_count_anomalies = abs(df['line_count'] - df['avg_line_count']) > \
            threshold_stddev * df['stddev_line_count']
            
        account_count_anomalies = abs(df['account_count'] - df['avg_account_count']) > \
            threshold_stddev * df['stddev_account_count']
            
        # Combine anomalies
        pattern_anomalies = df[line_count_anomalies | account_count_anomalies].copy()
        
        return pattern_anomalies
        
    def _analyze_anomaly_clustering(self, anomalies: pd.DataFrame) -> Dict[str, any]:
        """Analyze temporal clustering of anomalies"""
        if anomalies.empty:
            return {
                'cluster_count': 0,
                'max_cluster_size': 0,
                'avg_cluster_size': 0,
                'cluster_details': []
            }
            
        # Sort anomalies by date
        anomalies = anomalies.sort_values('journal_date')
        
        # Initialize clustering
        clusters = []
        current_cluster = []
        max_gap_days = 5  # Maximum days between anomalies to be considered in same cluster
        
        for idx in range(len(anomalies)):
            if idx == 0:
                current_cluster = [anomalies.iloc[idx]]
                continue
                
            date_diff = (anomalies.iloc[idx]['journal_date'] - 
                        anomalies.iloc[idx-1]['journal_date']).days
                        
            if date_diff <= max_gap_days:
                current_cluster.append(anomalies.iloc[idx])
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [anomalies.iloc[idx]]
                
        # Add last cluster
        if current_cluster:
            clusters.append(current_cluster)
            
        # Calculate cluster metrics
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        return {
            'cluster_count': len(clusters),
            'max_cluster_size': max(cluster_sizes) if clusters else 0,
            'avg_cluster_size': sum(cluster_sizes) / len(clusters) if clusters else 0,
            'cluster_details': [
                {
                    'start_date': cluster[0]['journal_date'].strftime('%Y-%m-%d'),
                    'end_date': cluster[-1]['journal_date'].strftime('%Y-%m-%d'),
                    'size': len(cluster),
                    'anomaly_types': list(set(a['type'] for a in cluster))
                }
                for cluster in clusters
            ]
        }
        
    def _calculate_anomaly_risk_metrics(self,
                                      anomaly_metrics: Dict,
                                      clustering_analysis: Dict) -> Dict[str, any]:
        """Calculate risk metrics based on anomaly analysis"""
        # Base risk on anomaly rate
        base_risk_score = min(anomaly_metrics['anomaly_rate'] * 2, 100)
        
        # Adjust for clustering
        cluster_factor = min(
            clustering_analysis['cluster_count'] * 
            clustering_analysis['avg_cluster_size'] / 10,
            1.0
        )
        
        # Calculate final risk score
        risk_score = base_risk_score * (1 + cluster_factor)
        
        # Determine risk level
        if risk_score <= 30:
            risk_level = 'LOW'
        elif risk_score <= 70:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
            
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_factors': {
                'anomaly_rate': anomaly_metrics['anomaly_rate'],
                'clustering_factor': round(cluster_factor * 100, 2),
                'amount_anomaly_count': anomaly_metrics['amount_anomalies'],
                'pattern_anomaly_count': anomaly_metrics['pattern_anomalies']
            }
        }
    
    def calculate_combined_risk_profile(self,
                                   account_id: int,
                                   lookback_days: int = 365) -> Dict[str, any]:
        """
        Calculate comprehensive risk profile combining multiple risk indicators
        
        Args:
            account_id: The GL account ID to analyze
            lookback_days: Number of days of historical data to analyze
            
        Returns:
            Dict containing comprehensive risk analysis
        """
        try:
            # Get volatility metrics
            volatility_analysis = self.analyze_account_volatility(
                account_id=account_id,
                period_months=lookback_days // 30
            )
            
            # Get transaction anomalies
            anomaly_analysis = self.analyze_transaction_anomalies(
                lookback_days=lookback_days
            )
            
            # Calculate trend metrics
            trend_analysis = self._analyze_risk_trends(
                account_id=account_id,
                lookback_days=lookback_days
            )
            
            # Combine risk metrics
            combined_risk_score = self._calculate_composite_risk_score(
                volatility_score=volatility_analysis.get('stability_metrics', {}).get('volatility_score', 0),
                anomaly_score=anomaly_analysis.get('risk_metrics', {}).get('risk_score', 0),
                trend_score=trend_analysis.get('trend_risk_score', 0)
            )
            
            # Get risk recommendations
            risk_recommendations = self._generate_risk_recommendations(
                volatility_analysis=volatility_analysis,
                anomaly_analysis=anomaly_analysis,
                trend_analysis=trend_analysis,
                combined_score=combined_risk_score
            )
            
            return {
                'status': 'ANALYZED',
                'account_id': account_id,
                'composite_risk': {
                    'risk_score': combined_risk_score,
                    'risk_level': self._determine_risk_level(combined_risk_score),
                    'confidence_level': self._calculate_confidence_level([
                        volatility_analysis,
                        anomaly_analysis,
                        trend_analysis
                    ])
                },
                'component_analysis': {
                    'volatility': volatility_analysis,
                    'anomalies': anomaly_analysis,
                    'trends': trend_analysis
                },
                'recommendations': risk_recommendations,
                'period': {
                    'start_date': (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'lookback_days': lookback_days
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating combined risk profile: {str(e)}")
            raise
            
    def _analyze_risk_trends(self,
                           account_id: int,
                           lookback_days: int) -> Dict[str, any]:
        """Analyze trends in risk indicators over time"""
        try:
            start_date = datetime.now() - timedelta(days=lookback_days)
            
            query = """
                WITH MonthlyMetrics AS (
                    SELECT 
                        DATE_TRUNC('month', j.journal_date) as month_start,
                        COUNT(DISTINCT j.journal_id) as transaction_count,
                        COUNT(DISTINCT l.account_id) as unique_accounts,
                        SUM(l.debit_amount) as total_debits,
                        SUM(l.credit_amount) as total_credits,
                        AVG(ABS(l.debit_amount - l.credit_amount)) as avg_imbalance,
                        MAX(GREATEST(l.debit_amount, l.credit_amount)) as max_amount
                    FROM GL_JOURNALS j
                    JOIN GL_JOURNAL_LINES l ON j.journal_id = l.journal_id
                    WHERE j.journal_date >= :start_date
                        AND l.account_id = :account_id
                    GROUP BY DATE_TRUNC('month', j.journal_date)
                    ORDER BY DATE_TRUNC('month', j.journal_date)
                )
                SELECT 
                    month_start,
                    transaction_count,
                    unique_accounts,
                    total_debits,
                    total_credits,
                    avg_imbalance,
                    max_amount,
                    AVG(transaction_count) OVER (
                        ORDER BY month_start 
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                    ) as avg_transaction_count,
                    AVG(max_amount) OVER (
                        ORDER BY month_start 
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                    ) as avg_max_amount
                FROM MonthlyMetrics
            """
            
            df = self.db.execute_query(
                query,
                params={
                    'account_id': account_id,
                    'start_date': start_date
                }
            )
            
            if df.empty:
                return {
                    'status': 'NO_DATA',
                    'trend_risk_score': 0
                }
                
            # Calculate trend metrics
            transaction_trend = stats.linregress(
                range(len(df)), 
                df['transaction_count']
            )
            
            amount_trend = stats.linregress(
                range(len(df)), 
                df['max_amount']
            )
            
            volatility_trend = stats.linregress(
                range(len(df)),
                df['avg_imbalance']
            )
            
            # Calculate trend scores
            trend_scores = {
                'transaction_volume': self._calculate_trend_score(transaction_trend.slope),
                'transaction_amount': self._calculate_trend_score(amount_trend.slope),
                'volatility': self._calculate_trend_score(volatility_trend.slope)
            }
            
            # Calculate overall trend risk score
            trend_risk_score = (
                trend_scores['transaction_volume'] * 0.3 +
                trend_scores['transaction_amount'] * 0.4 +
                trend_scores['volatility'] * 0.3
            )
            
            return {
                'status': 'ANALYZED',
                'trend_metrics': {
                    'transaction_volume_trend': float(transaction_trend.slope),
                    'amount_trend': float(amount_trend.slope),
                    'volatility_trend': float(volatility_trend.slope)
                },
                'trend_scores': trend_scores,
                'trend_risk_score': round(trend_risk_score * 100, 2),
                'trend_direction': self._determine_trend_direction(trend_risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk trends: {str(e)}")
            raise
            
    def _calculate_trend_score(self, slope: float) -> float:
        """Calculate normalized trend score"""
        # Convert slope to a 0-1 score where higher absolute values = higher risk
        return min(abs(slope) / 100, 1.0)
        
    def _determine_trend_direction(self, trend_score: float) -> str:
        """Determine trend direction based on score"""
        if trend_score < -0.1:
            return 'IMPROVING'
        elif trend_score > 0.1:
            return 'DETERIORATING'
        else:
            return 'STABLE'
            
    def _calculate_composite_risk_score(self,
                                      volatility_score: float,
                                      anomaly_score: float,
                                      trend_score: float) -> float:
        """Calculate composite risk score from components"""
        weights = {
            'volatility': 0.35,
            'anomalies': 0.35,
            'trend': 0.30
        }
        
        composite_score = (
            weights['volatility'] * volatility_score +
            weights['anomalies'] * anomaly_score +
            weights['trend'] * trend_score
        )
        
        return round(composite_score, 2)
        
    def _calculate_confidence_level(self, analyses: List[Dict]) -> str:
        """Calculate confidence level in risk assessment"""
        # Count analyses with sufficient data
        valid_analyses = sum(
            1 for analysis in analyses 
            if analysis.get('status') == 'ANALYZED'
        )
        
        if valid_analyses == len(analyses):
            return 'HIGH'
        elif valid_analyses >= len(analyses) // 2:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def _generate_risk_recommendations(self,
                                     volatility_analysis: Dict,
                                     anomaly_analysis: Dict,
                                     trend_analysis: Dict,
                                     combined_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Add recommendations based on composite risk score
        if combined_score >= 70:
            recommendations.extend([
                "Immediate review of account activity required",
                "Consider implementing additional controls",
                "Schedule detailed audit of recent transactions"
            ])
        elif combined_score >= 50:
            recommendations.extend([
                "Increase monitoring frequency",
                "Review unusual transaction patterns",
                "Validate recent significant changes"
            ])
            
        # Add specific recommendations based on component analyses
        if volatility_analysis.get('risk_level') == 'HIGH':
            recommendations.extend([
                "Review balance fluctuation patterns",
                "Consider implementing balance thresholds"
            ])
            
        if anomaly_analysis.get('risk_metrics', {}).get('risk_level') == 'HIGH':
            recommendations.extend([
                "Investigate clustered anomalies",
                "Review transaction approval processes"
            ])
            
        if trend_analysis.get('trend_direction') == 'DETERIORATING':
            recommendations.extend([
                "Analyze factors driving negative trends",
                "Develop risk mitigation strategy"
            ])
            
        return recommendations or ["Maintain current monitoring practices"]
            
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score <= 30:
            return 'LOW'
        elif risk_score <= 70:
            return 'MEDIUM'
        else:
            return 'HIGH'