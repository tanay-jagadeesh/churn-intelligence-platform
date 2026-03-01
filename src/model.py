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


# Risk thresholds: above 0.7 = high, 0.3-0.7 = medium, below 0.3 = low
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

RECOMMENDED_ACTIONS = {
    "High": "Offer discount or personal outreach from account manager",
    "Medium": "Send re-engagement email campaign with feature highlights",
    "Low": "No action needed — continue monitoring",
}


def segment_risk(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every customer with a churn probability and assign a risk tier.
    """
    X = features_df.drop(columns=["customer_id", "churned"])
    customer_ids = features_df["customer_id"]
    actual_churned = features_df["churned"].astype(int)

    churn_probs = model.predict_proba(X)[:, 1]

    tiers = []
    for prob in churn_probs:
        if prob >= HIGH_RISK_THRESHOLD:
            tiers.append("High")
        elif prob >= MEDIUM_RISK_THRESHOLD:
            tiers.append("Medium")
        else:
            tiers.append("Low")

    result = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": [round(p, 4) for p in churn_probs],
        "risk_tier": tiers,
        "recommended_action": [RECOMMENDED_ACTIONS[t] for t in tiers],
        "actually_churned": actual_churned,
    })

    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)