import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from data_processing import preprocess_pipeline, load_raw_data, drop_irrelevant_columns, fix_total_charges


# Path Setup 
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT / 'models'


# Load Model 
def load_model():
    """Load trained XGBoost model."""
    return joblib.load(MODELS_DIR / 'xgb_churn_model.pkl')


# Load Full Dataset 
def load_full_data():
    """
    Load and clean full dataset for reporting.
    Keeps Customer ID, Churn Reason, CLTV for business context.
    Does not drop leakage columns — reporting needs full picture.
    """
    df = load_raw_data()
    df = drop_irrelevant_columns(df)
    df = fix_total_charges(df)
    return df


# Generate Predictions
def generate_predictions(df, model, X_test, y_test):
    """
    Generate churn predictions and probabilities on test set.
    Attaches predictions back to original dataframe for reporting.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report_df = df.loc[y_test.index].copy()
    report_df['Predicted Churn'] = y_pred
    report_df['Churn Probability'] = y_prob.round(2)

    return report_df


# Churned Customers Report
def churned_customers_report(report_df):
    """
    Export list of customers predicted to churn.
    Includes Customer ID, Monthly Charges, Contract, Churn Reason, Probability.
    """
    churned = report_df[report_df['Predicted Churn'] == 1][[
        'CustomerID', 'Monthly Charges', 'Total Charges',
        'Contract', 'Payment Method', 'Churn Reason',
        'Churn Probability'
    ]].sort_values('Churn Probability', ascending=False)

    timestamp = datetime.now().strftime('%Y%m%d')
    path = REPORTS_DIR / f'churned_customers_{timestamp}.csv'
    churned.to_csv(path, index=False)
    print(f"Churned customers report saved: {path}")
    return churned


# High Risk Customers Report 
def high_risk_customers_report(report_df, threshold=0.7):
    """
    Export high risk customers — churn probability above threshold.
    Default threshold: 0.70
    """
    high_risk = report_df[report_df['Churn Probability'] >= threshold][[
        'CustomerID', 'Monthly Charges', 'Total Charges',
        'Contract', 'Tenure Months', 'Churn Probability'
    ]].sort_values('Churn Probability', ascending=False)

    timestamp = datetime.now().strftime('%Y%m%d')
    path = REPORTS_DIR / f'high_risk_customers_{timestamp}.csv'
    high_risk.to_csv(path, index=False)
    print(f"High risk customers report saved: {path}")
    return high_risk


# Summary Report 
def summary_report(report_df):
    """
    Generate summary statistics for business reporting.
    Includes churn rate, average charges, contract breakdown.
    """
    summary = {
        'Total Customers': int(len(report_df)),
        'Predicted Churners': int(report_df['Predicted Churn'].sum()),
        'Churn Rate (%)': float(round(report_df['Predicted Churn'].mean() * 100, 2)),
        'Avg Monthly Charges (Churned)': float(round(
            report_df[report_df['Predicted Churn'] == 1]['Monthly Charges'].mean(), 2)),
        'Avg Monthly Charges (Retained)': float(round(
            report_df[report_df['Predicted Churn'] == 0]['Monthly Charges'].mean(), 2)),
        'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    summary_df = pd.DataFrame([summary])
    timestamp = datetime.now().strftime('%Y%m%d')
    path = REPORTS_DIR / f'summary_report_{timestamp}.csv'
    summary_df.to_csv(path, index=False)
    print(f"Summary report saved: {path}")
    return summary


# Full Reporting Pipeline 
if __name__ == '__main__':
    # Load data and model
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    df = load_full_data()
    model = load_model()

    # Generate predictions
    report_df = generate_predictions(df, model, X_test, y_test)

    # Generate reports
    churned = churned_customers_report(report_df)
    high_risk = high_risk_customers_report(report_df)
    summary = summary_report(report_df)

    print("\nAll reports generated successfully")
    print(f"Churned customers: {len(churned)}")
    print(f"High risk customers: {len(high_risk)}")
    print(f"Summary: {summary}")