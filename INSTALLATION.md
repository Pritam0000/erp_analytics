# Installation Guide

## Prerequisites
- Python 3.11+
- Oracle Client (if connecting to Oracle DB)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter network issues, try:
```bash
pip install --no-cache-dir -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib plotly
```

## Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials
```

## Step 3: Generate Sample Data (Already Done)

Sample data is already in `data/sample_invoices.csv` and `data/sample_suppliers.csv`

## Step 4: Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Troubleshooting

### ModuleNotFoundError
If you see "ModuleNotFoundError: No module named 'pandas'", install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```

### Oracle Connection Issues
Set `USE_MOCK_DB=true` in `.env` to use CSV files instead of Oracle database.
