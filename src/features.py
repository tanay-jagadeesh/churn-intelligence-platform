"""
Feature engineering: transforms raw monthly activity into one row per customer
with summary statistics that the ML model can learn from.
"""

import numpy as np
import pandas as pd


def _compute_trend(series: pd.Series) -> float:
  
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
  
    slope, _ = np.polyfit(x, series.values, 1)
    return round(slope, 4)


def build_features(customers_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:

    features = []

    for customer_id, group in activity_df.groupby("customer_id"):
        group = group.sort_values("month")

   
        avg_logins = group["logins"].mean()
        avg_sessions = group["total_sessions"].mean()
        avg_duration = group["avg_session_duration"].mean()
        avg_feature_usage = group["feature_usage_score"].mean()
        avg_satisfaction = group["satisfaction_score"].mean()

        recent = group.tail(3)
        recent_avg_logins = recent["logins"].mean()
        recent_avg_sessions = recent["total_sessions"].mean()
        recent_avg_satisfaction = recent["satisfaction_score"].mean()

        login_trend = _compute_trend(group["logins"])
        session_trend = _compute_trend(group["total_sessions"])
        satisfaction_trend = _compute_trend(group["satisfaction_score"])

        total_tickets = group["support_tickets"].sum()
        tenure_months = len(group)

        ontime_rate = (group["payment_status"] == "On-time").mean()

        features.append({
            "customer_id": customer_id,
            "avg_logins": round(avg_logins, 2),
            "avg_sessions": round(avg_sessions, 2),
            "avg_duration": round(avg_duration, 2),
            "avg_feature_usage": round(avg_feature_usage, 2),
            "avg_satisfaction": round(avg_satisfaction, 2),
            "recent_avg_logins": round(recent_avg_logins, 2),
            "recent_avg_sessions": round(recent_avg_sessions, 2),
            "recent_avg_satisfaction": round(recent_avg_satisfaction, 2),
            "login_trend": login_trend,
            "session_trend": session_trend,
            "satisfaction_trend": satisfaction_trend,
            "total_support_tickets": total_tickets,
            "tenure_months": tenure_months,
            "ontime_payment_rate": round(ontime_rate, 4),
        })

    features_df = pd.DataFrame(features)

    customer_cols = [
        "customer_id", "age", "gender", "plan_type", "monthly_charge",
        "acquisition_channel", "contract_type", "churned",
    ]
    merged = features_df.merge(customers_df[customer_cols], on="customer_id", how="left")

    merged = pd.get_dummies(
        merged,
        columns=["gender", "plan_type", "acquisition_channel", "contract_type"],
        drop_first=True,
    )

    return merged