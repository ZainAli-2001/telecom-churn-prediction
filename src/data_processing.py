import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Path Setup
ROOT = Path(__file__).resolve().parent.parent


# Data Loading
def load_raw_data():
    """Load raw dataset from Excel file."""
    return pd.read_excel(ROOT / 'data' / 'raw' / 'Telco_customer_churn.xlsx')


# Cleaning 
def drop_irrelevant_columns(df):
    """
    Drop non-informative columns identified during EDA:
    - Constant value columns: Count, Country, State
    - Geographic columns not useful for modeling: Zip Code, Lat Long, Latitude, Longitude
    """
    cols_to_drop = [
        'Count', 'Country', 'State', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude'
    ]
    return df.drop(columns=cols_to_drop)


def fix_total_charges(df):
    """
    Total Charges column contained blank spaces discovered during EDA.
    Converted to numeric and filled with 0 for customers with Tenure Month = 0.
    """
    df = df.copy()
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(0)
    return df


def drop_leakage_and_redundant_columns(df):
    """
    Drop columns before modeling:
    - CustomerID: unique identifier, not a predictive feature
    - Churn Label: redundant, Churn Value (0/1) used as target variable
    - Churn Score: derived metric, causes data leakage
    - CLTV: derived metric, causes data leakage
    - Churn Reason: not available at prediction time, causes leakage
    - City: 1129 unique cities, high cardinality, no meaningful impact on model
    """
    cols_to_drop = [
        'CustomerID', 'Churn Label', 'Churn Score',
        'CLTV', 'Churn Reason', 'City'
    ]
    return df.drop(columns=cols_to_drop)


# Encoding 
BINARY_COLS = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Paperless Billing', 'Online Security',
    'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Multiple Lines'
]

MULTI_COLS = [
    'Internet Service', 'Contract', 'Payment Method'
]


def encode_features(X_train, X_test):
    """
    Encode categorical features:
    - Label Encoding for binary columns (Yes/No)
    - One Hot Encoding for multi-category columns
    - fit on X_train only to prevent data leakage
    """
    le = LabelEncoder()

    # Label Encoding
    for col in BINARY_COLS:
        if col in X_train.columns:
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    # One Hot Encoding
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test


# Train Test Split
def split_data(df):
    """
    Split data into train and test sets.
    - Stratified to preserve 74/26 churn ratio
    - 80/20 split
    """
    X = df.drop('Churn Value', axis=1)
    y = df['Churn Value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


# Full Pipeline
def preprocess_pipeline():
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Drop irrelevant columns
    3. Fix Total Charges
    4. Drop leakage and redundant columns
    5. Split data
    6. Encode features
    """
    df = load_raw_data()
    df = drop_irrelevant_columns(df)
    df = fix_total_charges(df)
    df = drop_leakage_and_redundant_columns(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = encode_features(X_train, X_test)

    return X_train, X_test, y_train, y_test

# Upload Preprocessing 
def preprocess_uploaded_data(df, X_train_columns):
    df = drop_irrelevant_columns(df) if 'Count' in df.columns else df
    df = fix_total_charges(df)
    df = drop_leakage_and_redundant_columns(df)
    
    le = LabelEncoder()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=X_train_columns, fill_value=0)
    
    return df

def load_full_data():
    """
    Load and clean full dataset for reporting.
    Keeps CustomerID, Churn Reason, CLTV for business context.
    """
    df = load_raw_data()
    df = drop_irrelevant_columns(df)
    df = fix_total_charges(df)
    return df

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    print("Preprocessing complete")
    print(f"Training features: {X_train.shape}")
    print(f"Testing features : {X_test.shape}")
    print(f"Training labels  : {y_train.shape}")
    print(f"Testing labels   : {y_test.shape}")