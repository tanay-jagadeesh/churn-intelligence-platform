"""
Main entry point for the Churn Intelligence Platform.

"""

import argparse

from src.generators import generate_customers, generate_monthly_activity
from src.features import build_features
from src.model import train_model, evaluate_model, get_feature_importance, segment_risk
from src.utils import save_csv, load_csv, save_model


def generate_data():

    print("Generating customer data...")
    customers = generate_customers()
    save_csv(customers, "customers.csv")
    print(f"  Saved {len(customers)} customers to data/customers.csv")

    churn_count = customers["churned"].sum()
    print(f"  Churn rate: {churn_count}/{len(customers)} ({churn_count/len(customers):.1%})")

    print("Generating monthly activity data")
    activity = generate_monthly_activity(customers)
    save_csv(activity, "monthly_activity.csv")
    print(f"  Saved {len(activity)} activity records to data/monthly_activity.csv")

    return customers, activity


def train_and_evaluate():

    print("\nLoading data")
    customers = load_csv("customers.csv")
    activity = load_csv("monthly_activity.csv")

    print("Building features")
    features = build_features(customers, activity)
    save_csv(features, "features.csv")
    print(f"  Created {len(features)} feature rows with {len(features.columns)} columns")

    print("Training model")
    model, X_test, y_test = train_model(features)
    save_model(model, "churn_model.joblib")
    print("  Model saved to models/churn_model.joblib")

    print("\n Evaluation Results")
    results = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"\n{results['report_text']}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    print("\nTop 10 Most Important Features")
    importance = get_feature_importance(model, X_test.columns.tolist())
    print(importance.head(10).to_string(index=False))

    print("\n Risk Segmentation")
    risk_df = segment_risk(model, features)
    save_csv(risk_df, "risk_segments.csv")

    tier_counts = risk_df["risk_tier"].value_counts()
    for tier in ["High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(risk_df)
        print(f"  {tier:6s}: {count:4d} customers ({pct:.1%})")

    print("Top 10 Highest-Risk Customers:")
    top10 = risk_df.head(10)[["customer_id", "churn_probability", "risk_tier", "recommended_action"]]
    print(top10.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Churn Intelligence Platform")
    parser.add_argument("--generate", action="store_true", help="Only generate data")
    parser.add_argument("--train", action="store_true", help="Only train and evaluate")
    args = parser.parse_args()

    if args.generate:
        generate_data()
    elif args.train:
        train_and_evaluate()
    else:
   
        generate_data()
        train_and_evaluate()

    print("\nDone.")


if __name__ == "__main__":
    main()