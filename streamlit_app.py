# streamlit_app.py
"""
Invoice Predictive Analytics & Anomaly Detection System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Invoice Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .warning-text {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-text {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize services (cached to avoid reloading)
@st.cache_resource
def init_services():
    """Initialize data loader and services"""
    try:
        from src.data.data_loader import InvoiceDataLoader
        from src.services.prediction_service import PredictionService
        from src.services.anomaly_service import AnomalyService

        data_loader = InvoiceDataLoader(use_mock_db=True)
        prediction_service = PredictionService(data_loader)
        anomaly_service = AnomalyService(data_loader)

        return data_loader, prediction_service, anomaly_service
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        st.info("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
        return None, None, None


# Load data (cached)
@st.cache_data
def load_invoice_data():
    """Load invoice data"""
    try:
        data_loader, _, _ = init_services()
        if data_loader is None:
            return None, None

        invoices_df = data_loader.load_invoices()
        suppliers_df = data_loader.load_suppliers()
        return invoices_df, suppliers_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


# Navigation
def main():
    """Main application"""

    # Sidebar navigation
    st.sidebar.image("https://via.placeholder.com/150x50?text=Invoice+Analytics", use_column_width=True)
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home Dashboard", "🔮 Predictive Analytics", "🚨 Anomaly Detection", "📊 Model Performance"]
    )

    # Load data
    invoices_df, suppliers_df = load_invoice_data()

    if invoices_df is None:
        st.error("Failed to load data. Please check the installation guide.")
        return

    # Route to pages
    if page == "🏠 Home Dashboard":
        show_home_dashboard(invoices_df, suppliers_df)
    elif page == "🔮 Predictive Analytics":
        show_predictive_analytics(invoices_df, suppliers_df)
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(invoices_df, suppliers_df)
    elif page == "📊 Model Performance":
        show_model_performance(invoices_df, suppliers_df)


def show_home_dashboard(invoices_df, suppliers_df):
    """Home dashboard with KPIs and charts"""
    st.markdown('<p class="main-header">📊 Invoice Analytics Dashboard</p>', unsafe_allow_html=True)

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Invoices",
            value=f"{len(invoices_df):,}",
            delta=f"+{len(invoices_df[invoices_df['invoice_status'] == 'PENDING'])} Pending"
        )

    with col2:
        total_amount = invoices_df['invoice_amount'].sum()
        st.metric(
            label="Total Amount",
            value=f"${total_amount:,.2f}",
            delta=f"Avg: ${invoices_df['invoice_amount'].mean():,.2f}"
        )

    with col3:
        rejection_rate = (invoices_df['invoice_status'] == 'REJECTED').mean() * 100
        st.metric(
            label="Rejection Rate",
            value=f"{rejection_rate:.1f}%",
            delta=f"{(invoices_df['invoice_status'] == 'REJECTED').sum()} Rejected",
            delta_color="inverse"
        )

    with col4:
        supplier_count = invoices_df['supplier_id'].nunique()
        st.metric(
            label="Active Suppliers",
            value=f"{supplier_count}",
            delta=f"Top: {invoices_df.groupby('supplier_name').size().idxmax()}"
        )

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Invoice Status Distribution")
        status_counts = invoices_df['invoice_status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Invoices by Status",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Invoice Category Distribution")
        category_counts = invoices_df['invoice_category'].value_counts().head(8)
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Top 8 Invoice Categories",
            labels={'x': 'Category', 'y': 'Count'},
            color=category_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    st.subheader("Invoice Trends Over Time")
    invoices_df['invoice_date'] = pd.to_datetime(invoices_df['invoice_date'])
    daily_invoices = invoices_df.groupby(invoices_df['invoice_date'].dt.date).size().reset_index()
    daily_invoices.columns = ['Date', 'Count']

    fig = px.line(
        daily_invoices,
        x='Date',
        y='Count',
        title="Daily Invoice Volume",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top suppliers table
    st.subheader("Top 10 Suppliers by Invoice Count")
    top_suppliers = invoices_df.groupby('supplier_name').agg({
        'invoice_id': 'count',
        'invoice_amount': ['sum', 'mean']
    }).round(2)
    top_suppliers.columns = ['Invoice Count', 'Total Amount', 'Avg Amount']
    top_suppliers = top_suppliers.sort_values('Invoice Count', ascending=False).head(10)

    st.dataframe(top_suppliers, use_container_width=True)

    # Recent invoices
    st.subheader("Recent Invoices")
    recent_invoices = invoices_df.sort_values('created_date', ascending=False).head(10)
    display_cols = ['invoice_number', 'supplier_name', 'invoice_date', 'invoice_amount', 'invoice_status']
    st.dataframe(recent_invoices[display_cols], use_container_width=True)


def show_predictive_analytics(invoices_df, suppliers_df):
    """Predictive analytics page"""
    st.markdown('<p class="main-header">🔮 Predictive Analytics</p>', unsafe_allow_html=True)

    _, prediction_service, _ = init_services()

    if prediction_service is None:
        st.error("Prediction service not available")
        return

    # Train models button
    if not prediction_service.is_trained:
        st.warning("⚠️ Models are not trained yet. Click below to train models.")

        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a minute."):
                try:
                    results = prediction_service.train_all_models()
                    st.success("✅ All models trained successfully!")
                    st.json(results)
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    st.info("Note: This is expected if sklearn/xgboost are not installed.")
        return

    # Tabs for different predictions
    tab1, tab2, tab3 = st.tabs([
        "💰 Amount Prediction",
        "⏱️ Payment Delay Prediction",
        "❌ Rejection Probability"
    ])

    with tab1:
        st.subheader("Invoice Amount Prediction")
        st.write("Predict expected invoice amounts based on supplier history and PO.")

        # Sample prediction
        sample_invoice = invoices_df.sample(5)

        try:
            predictions = prediction_service.predict_invoice_amount(sample_invoice, 'xgboost')

            results_df = pd.DataFrame({
                'Invoice Number': sample_invoice['invoice_number'].values,
                'Actual Amount': sample_invoice['invoice_amount'].values,
                'Predicted Amount': predictions['predicted_amount'].values,
                'Lower Bound': predictions['lower_bound'].values,
                'Upper Bound': predictions['upper_bound'].values
            })

            st.dataframe(results_df, use_container_width=True)

            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=results_df['Actual Amount'],
                mode='markers',
                name='Actual',
                marker=dict(size=10, color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=results_df['Predicted Amount'],
                mode='markers',
                name='Predicted',
                marker=dict(size=10, color='red')
            ))
            fig.update_layout(title="Actual vs Predicted Amounts", xaxis_title="Sample", yaxis_title="Amount ($)")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    with tab2:
        st.subheader("Payment Delay Prediction")
        st.write("Forecast how many days after the due date payment will be received.")

        # Filter to pending/approved invoices
        pending_invoices = invoices_df[invoices_df['invoice_status'].isin(['PENDING', 'APPROVED'])].head(10)

        if len(pending_invoices) > 0:
            try:
                delay_predictions = prediction_service.predict_payment_delay(pending_invoices, 'xgboost')

                results_df = pd.DataFrame({
                    'Invoice Number': pending_invoices['invoice_number'].values,
                    'Due Date': pending_invoices['due_date'].values,
                    'Predicted Delay (Days)': delay_predictions['predicted_delay_days'].values,
                    'Expected Payment Date': delay_predictions['predicted_payment_date'].values,
                    'Risk Level': delay_predictions['risk_level'].values
                })

                st.dataframe(results_df, use_container_width=True)

                # Risk distribution
                risk_counts = delay_predictions['risk_level'].value_counts()
                fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                             title="Payment Delay Risk Distribution",
                             labels={'x': 'Risk Level', 'y': 'Count'},
                             color=risk_counts.values)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.info("No pending/approved invoices to predict")

    with tab3:
        st.subheader("Invoice Rejection Probability")
        st.write("Calculate the likelihood of invoice rejection.")

        sample_invoices = invoices_df.sample(10)

        try:
            rejection_predictions = prediction_service.predict_rejection_probability(sample_invoices, 'xgboost')

            results_df = pd.DataFrame({
                'Invoice Number': sample_invoices['invoice_number'].values,
                'Amount': sample_invoices['invoice_amount'].values,
                'Rejection Probability': rejection_predictions['rejection_probability'].values,
                'Risk Level': rejection_predictions['risk_level'].values,
                'Recommendation': rejection_predictions['recommendation'].values
            })

            # Color code by risk
            def color_risk(val):
                if val == 'High Risk':
                    return 'background-color: #ffcccc'
                elif val == 'Medium Risk':
                    return 'background-color: #fff4cc'
                else:
                    return 'background-color: #ccffcc'

            st.dataframe(
                results_df.style.applymap(color_risk, subset=['Risk Level']),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def show_anomaly_detection(invoices_df, suppliers_df):
    """Anomaly detection page"""
    st.markdown('<p class="main-header">🚨 Anomaly Detection</p>', unsafe_allow_html=True)

    _, _, anomaly_service = init_services()

    if anomaly_service is None:
        st.error("Anomaly service not available")
        return

    # Run detection button
    if st.button("🔍 Run All Anomaly Detections", type="primary"):
        with st.spinner("Detecting anomalies... This may take a moment."):
            try:
                results = anomaly_service.detect_all_anomalies()
                st.session_state['anomaly_results'] = results
                st.success("✅ Anomaly detection complete!")
            except Exception as e:
                st.error(f"Error detecting anomalies: {str(e)}")
                st.info("Note: This is expected if sklearn is not installed.")

    # Show results if available
    if 'anomaly_results' in st.session_state:
        results = st.session_state['anomaly_results']

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            dup_count = len(results.get('duplicates', []))
            st.metric("Duplicate Invoices", dup_count)

        with col2:
            amt_count = len(results.get('amount_anomalies', []))
            st.metric("Amount Anomalies", amt_count)

        with col3:
            po_count = len(results.get('po_mismatches', []))
            st.metric("PO Mismatches", po_count)

        st.divider()

        # Tabs for different anomalies
        tab1, tab2, tab3 = st.tabs([
            "🔄 Duplicate Invoices",
            "💵 Amount Anomalies",
            "📄 PO Mismatches"
        ])

        with tab1:
            st.subheader("Duplicate Invoice Detection")
            duplicates_df = results.get('duplicates', pd.DataFrame())

            if len(duplicates_df) > 0:
                st.write(f"Found **{len(duplicates_df)}** potential duplicates")

                # Show table
                display_cols = ['invoice_number', 'supplier_name', 'invoice_amount',
                                'duplicate_type', 'duplicate_score', 'severity']
                display_cols = [col for col in display_cols if col in duplicates_df.columns]
                st.dataframe(duplicates_df[display_cols], use_container_width=True)

                # Chart
                if 'duplicate_type' in duplicates_df.columns:
                    type_counts = duplicates_df['duplicate_type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index,
                                 title="Duplicates by Detection Method")
                    st.plotly_chart(fig)
            else:
                st.success("✅ No duplicates detected!")

        with tab2:
            st.subheader("Amount Anomaly Detection")
            amount_anomalies_df = results.get('amount_anomalies', pd.DataFrame())

            if len(amount_anomalies_df) > 0:
                st.write(f"Found **{len(amount_anomalies_df)}** amount anomalies")

                display_cols = ['invoice_number', 'supplier_name', 'invoice_amount',
                                'supplier_avg_amount', 'detection_methods', 'severity']
                display_cols = [col for col in display_cols if col in amount_anomalies_df.columns]
                st.dataframe(amount_anomalies_df[display_cols], use_container_width=True)
            else:
                st.success("✅ No amount anomalies detected!")

        with tab3:
            st.subheader("PO-Invoice Mismatch Detection")
            po_mismatches_df = results.get('po_mismatches', pd.DataFrame())

            if len(po_mismatches_df) > 0:
                st.write(f"Found **{len(po_mismatches_df)}** PO mismatches")

                display_cols = ['invoice_number', 'supplier_name', 'invoice_amount', 'po_amount',
                                'mismatch_type', 'severity', 'recommendation']
                display_cols = [col for col in display_cols if col in po_mismatches_df.columns]
                st.dataframe(po_mismatches_df[display_cols], use_container_width=True)

                # PO Compliance
                po_compliance = results.get('po_compliance', {})
                if po_compliance:
                    st.subheader("PO Compliance Analysis")
                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.metric("PO Coverage Rate",
                                  f"{po_compliance.get('po_coverage_rate', 0) * 100:.1f}%")

                    with comp_col2:
                        st.metric("High-Value PO Rate",
                                  f"{po_compliance.get('high_value_po_rate', 0) * 100:.1f}%")
            else:
                st.success("✅ No PO mismatches detected!")


def show_model_performance(invoices_df, suppliers_df):
    """Model performance page"""
    st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)

    _, prediction_service, _ = init_services()

    if prediction_service is None:
        st.error("Prediction service not available")
        return

    if not prediction_service.is_trained:
        st.warning("Models not trained yet. Please train models in the Predictive Analytics page.")
        return

    st.subheader("Model Comparison: Random Forest vs XGBoost")

    # Placeholder metrics (would be from actual training)
    st.info("Train models in Predictive Analytics page to see performance metrics here.")

    # Feature importance (if models trained)
    st.subheader("Feature Importance")

    try:
        importance_df = prediction_service.amount_predictor.get_feature_importance('xgboost', top_n=10)

        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Important Features (Amount Predictor - XGBoost)",
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info("Train models to see feature importance")


# Run the app
if __name__ == "__main__":
    main()
