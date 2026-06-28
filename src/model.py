import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from data_processing import preprocess_pipeline


# Path Setup
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


# Model Training
def train_model(X_train, y_train):
    """
    Train XGBoost model with best parameters identified from GridSearchCV.
    - scale_pos_weight=2.77 handles class imbalance (5174/1869)
    - max_depth=3 prevents overfitting
    - learning_rate=0.1 for stable convergence
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=2.77,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    Prints classification report with precision, recall, F1.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


# Model Saving
def save_model(model):
    """Save trained model to models directory."""
    joblib.dump(model, MODELS_DIR / 'xgb_churn_model.pkl')
    print("Model saved successfully")


# Model Loading 
def load_model():
    """Load trained model from models directory."""
    return joblib.load(MODELS_DIR / 'xgb_churn_model.pkl')


# Prediction 
def predict(model, X):
    """
    Generate churn predictions and probabilities.
    Returns both binary predictions and churn probability scores.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


# Full Pipeline
if __name__ == '__main__':
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_pipeline()

    # Train model
    print("Training XGBoost model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    # Save training column names for upload alignment
    joblib.dump(list(X_train.columns), MODELS_DIR / 'model_columns.pkl')
    print("Model columns saved")