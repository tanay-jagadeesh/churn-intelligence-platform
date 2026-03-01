"""
Churn prediction model: training, evaluation, and feature importance.

Uses a Random Forest classifier. The workflow:
  1. Split data into train (80%) and test (20%)
  2. Train the model on the training set
  3. Evaluate on the test set (accuracy, precision, recall, F1)
  4. Report which features are most important for predicting churn
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config.schema import MODEL_PARAMS, RANDOM_SEED, TEST_SIZE


def train_model(features_df: pd.DataFrame):
 
    X = features_df.drop(columns=["customer_id", "churned"])
    y = features_df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
  
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = report["accuracy"]

    return {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "report_text": classification_report(y_test, y_pred),
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)