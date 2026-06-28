import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Path Setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / 'src'))

from data_processing import (
    preprocess_pipeline,
    load_full_data,
    preprocess_uploaded_data,
    drop_irrelevant_columns,
    fix_total_charges,
    drop_leakage_and_redundant_columns,
    encode_features,
    BINARY_COLS
)

MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# Page Config
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model and Columns
@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / 'xgb_churn_model.pkl')
    columns = joblib.load(MODELS_DIR / 'model_columns.pkl')
    return model, columns

# Load and Preprocess Data
@st.cache_data
def load_data():
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    df = load_full_data()
    return X_train, X_test, y_train, y_test, df

model, model_columns = load_model()
X_train, X_test, y_train, y_test, df = load_data()

# Generate and Cache Predictions on Full Dataset
@st.cache_data
def get_predictions():
    df_full = load_full_data()
    df_processed = preprocess_uploaded_data(df_full.copy(), model_columns)
    y_pred_full = model.predict(df_processed)
    y_prob_full = model.predict_proba(df_processed)[:, 1]
    df_full['Predicted Churn'] = y_pred_full
    df_full['Churn Probability'] = y_prob_full.round(2)
    return df_full

report_df = get_predictions()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", [
    "Overview",
    "EDA Insights",
    "Churn Predictions",
    "Upload & Predict",
    "Download Reports"
])

# Overview Page
if page == "Overview":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("---")

    total = len(report_df)
    churned = int(report_df['Predicted Churn'].sum())
    retained = total - churned
    churn_rate = round(report_df['Predicted Churn'].mean() * 100, 2)
    avg_monthly_churned = round(report_df[report_df['Predicted Churn'] == 1]['Monthly Charges'].mean(), 2)
    avg_monthly_retained = round(report_df[report_df['Predicted Churn'] == 0]['Monthly Charges'].mean(), 2)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Customers", total)
    col2.metric("Predicted Churners", churned)
    col3.metric("Retained Customers", retained)
    col4.metric("Churn Rate", f"{churn_rate}%")
    col5.metric("Avg Monthly (Churned)", f"${avg_monthly_churned}")
    col6.metric("Avg Monthly (Retained)", f"${avg_monthly_retained}")

    st.info("Note: Predicted churn rate may differ from actual due to model optimized for high recall — minimizing missed churners.")

    st.markdown("---")
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    report_df['Predicted Churn'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
    ax.set_xticklabels(['Retained', 'Churned'], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Predicted Churn Distribution")
    st.pyplot(fig)

# EDA Insights Page
elif page == "EDA Insights":
    st.title("📈 EDA Insights")
    st.markdown("---")

    # Map numeric predictions to readable labels
    report_df['Churn Status'] = report_df['Predicted Churn'].map({0: 'Retained', 1: 'Churned'})
    
    st.subheader("Monthly Charges by Churn Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=report_df, x='Churn Status', y='Monthly Charges', ax=ax)
    ax.set_xticklabels(['Retained', 'Churned'])
    ax.set_title("Monthly Charges by Churn Status")
    st.pyplot(fig)

    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(data=report_df, x='Contract', hue='Churn Status', ax=ax)
    ax.set_title("Churn Distribution by Contract Type")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Tenure Months by Churn Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=report_df, x='Churn Status', y='Tenure Months', ax=ax)
    ax.set_xticklabels(['Retained', 'Churned'])
    ax.set_title("Tenure Months by Churn Status")
    st.pyplot(fig)

    st.subheader("Churn by Internet Service")
    fig, ax = plt.subplots()
    sns.countplot(data=report_df, x='Internet Service', hue='Churn Status', ax=ax)
    ax.set_title("Churn Distribution by Internet Service")
    st.pyplot(fig)

# Churn Predictions Page
elif page == "Churn Predictions":
    st.title("🔮 Churn Predictions")
    st.markdown("---")

    st.subheader("High Risk Customers (Churn Probability ≥ 0.70)")
    high_risk = report_df[report_df['Churn Probability'] >= 0.70][[
        'CustomerID', 'Monthly Charges', 'Total Charges',
        'Contract', 'Tenure Months', 'Churn Probability'
    ]].sort_values('Churn Probability', ascending=False)
    st.dataframe(high_risk, use_container_width=True)

    st.subheader("All Predicted Churners")
    churned_df = report_df[report_df['Predicted Churn'] == 1][[
        'CustomerID', 'Monthly Charges', 'Contract',
        'Payment Method', 'Churn Reason', 'Churn Probability'
    ]].sort_values('Churn Probability', ascending=False)
    st.dataframe(churned_df, use_container_width=True)

# Upload and Predict Page
elif page == "Upload & Predict":
    st.title("📂 Upload & Predict")
    st.markdown("---")
    st.info("Upload a CSV or Excel file in the same format as the IBM Telco dataset for churn predictions.")

    uploaded_file = st.file_uploader("Upload customer data", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        timestamp = datetime.now().strftime('%Y%m%d')
        try:
            if uploaded_file.name.endswith('.csv'):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            st.success(f"File uploaded successfully — {len(uploaded_df)} rows detected")

            # Validate columns
            REQUIRED_COLS = ['Tenure Months', 'Monthly Charges', 'Total Charges',
                           'Contract', 'Payment Method', 'Internet Service',
                           'Online Security', 'Tech Support']
            missing_cols = [col for col in REQUIRED_COLS if col not in uploaded_df.columns]
            if missing_cols:
                st.warning(f"Missing important columns: {missing_cols} — predictions may be inaccurate")

            # Preprocess and predict
            processed = preprocess_uploaded_data(uploaded_df.copy(), model_columns)
            predictions = model.predict(processed)
            probabilities = model.predict_proba(processed)[:, 1].round(2)

            uploaded_df['Predicted Churn'] = predictions
            uploaded_df['Churn Probability'] = probabilities

            st.subheader("Prediction Results")
            st.dataframe(uploaded_df[['CustomerID', 'Predicted Churn', 'Churn Probability']]
                        if 'CustomerID' in uploaded_df.columns
                        else uploaded_df[['Predicted Churn', 'Churn Probability']],
                        use_container_width=True)

            churned_count = int(predictions.sum())
            st.metric("Predicted Churners", churned_count)
            st.metric("Churn Rate", f"{round(predictions.mean() * 100, 2)}%")

            st.download_button(
                label="Download Prediction Results CSV",
                data=uploaded_df.to_csv(index=False),
                file_name=f"upload_predictions_{timestamp}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Download Reports Page
elif page == "Download Reports":
    st.title("📥 Download Reports")
    st.markdown("---")

    timestamp = datetime.now().strftime('%Y%m%d')

    # Churned Customers Report
    st.subheader("Churned Customers Report")
    churned_report = report_df[report_df['Predicted Churn'] == 1][[
        'CustomerID', 'Monthly Charges', 'Total Charges',
        'Contract', 'Payment Method', 'Churn Reason', 'Churn Probability'
    ]].sort_values('Churn Probability', ascending=False)

    st.download_button(
        label="Download Churned Customers CSV",
        data=churned_report.to_csv(index=False),
        file_name=f"churned_customers_{timestamp}.csv",
        mime="text/csv"
    )

    # High Risk Customers Report
    st.subheader("High Risk Customers Report")
    high_risk_report = report_df[report_df['Churn Probability'] >= 0.70][[
        'CustomerID', 'Monthly Charges', 'Total Charges',
        'Contract', 'Tenure Months', 'Churn Probability'
    ]].sort_values('Churn Probability', ascending=False)

    st.download_button(
        label="Download High Risk Customers CSV",
        data=high_risk_report.to_csv(index=False),
        file_name=f"high_risk_customers_{timestamp}.csv",
        mime="text/csv"
    )

    # Summary Report
    st.subheader("Summary Report")
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

    st.download_button(
        label="Download Summary Report CSV",
        data=summary_df.to_csv(index=False),
        file_name=f"summary_report_{timestamp}.csv",
        mime="text/csv"
    )