"""
Synthetic data generators for customers and monthly activity.
"""

import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from config.schema import (
    ACQ_CHANNELS,
    ACQ_WEIGHTS,
    AGE_MAX,
    AGE_MEAN,
    AGE_MIN,
    AGE_STD,
    BASELINE_FEATURE_USAGE,
    BASELINE_LOGINS,
    BASELINE_SATISFACTION,
    BASELINE_SESSION_DURATION,
    BASELINE_SESSIONS,
    CHURN_RATE,
    CONTRACT_TYPES,
    CONTRACT_WEIGHTS,
    GENDERS,
    GENDER_WEIGHTS,
    LOCATIONS,
    NUM_CUSTOMERS,
    PAYMENT_STATUSES,
    PLAN_PRICES,
    PLAN_TYPES,
    RANDOM_SEED,
)

fake = Faker()
Faker.seed(RANDOM_SEED)


def generate_customers(num_customers: int = NUM_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    """
    rng = np.random.default_rng(seed)
    today = date.today()
    two_years_ago = today - timedelta(days=730)

    customers = []
    for _ in range(num_customers):
  
        age = int(np.clip(rng.normal(AGE_MEAN, AGE_STD), AGE_MIN, AGE_MAX))

        days_since_signup = rng.integers(0, 730)
        signup_date = two_years_ago + timedelta(days=int(days_since_signup))

        plan = rng.choice(PLAN_TYPES)
        price_range = PLAN_PRICES[plan]
        monthly_charge = round(rng.uniform(price_range[0], price_range[1]), 2)

        contract = rng.choice(CONTRACT_TYPES, p=CONTRACT_WEIGHTS)
        churn_prob = CHURN_RATE * (0.5 if contract == "Annual" else 1.2)
        churned = bool(rng.random() < churn_prob)

        churn_date = None
        if churned:
            days_active = (today - signup_date).days
            if days_active > 30:
                churn_offset = rng.integers(30, max(31, days_active))
                churn_date = signup_date + timedelta(days=int(churn_offset))
            else:
                churned = False

        gender = rng.choice(GENDERS, p=GENDER_WEIGHTS)

        customers.append({
            "customer_id": str(uuid.uuid4())[:12],
            "name": fake.name(),
            "email": fake.email(),
            "age": age,
            "gender": gender,
            "location": rng.choice(LOCATIONS),
            "signup_date": signup_date,
            "plan_type": plan,
            "monthly_charge": monthly_charge,
            "acquisition_channel": rng.choice(ACQ_CHANNELS, p=ACQ_WEIGHTS),
            "contract_type": contract,
            "churned": churned,
            "churn_date": churn_date,
        })

    return pd.DataFrame(customers)


def generate_monthly_activity(customers_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate monthly behavioral data for each customer.
    """
    rng = np.random.default_rng(seed)
    today = date.today()
    records = []

    for _, cust in customers_df.iterrows():
        signup = cust["signup_date"]
        end = cust["churn_date"] if cust["churned"] else today
        if pd.isna(end):
            end = today

        base_logins = rng.uniform(*BASELINE_LOGINS)
        base_sessions = rng.uniform(*BASELINE_SESSIONS)
        base_duration = rng.uniform(*BASELINE_SESSION_DURATION)
        base_feature = rng.uniform(*BASELINE_FEATURE_USAGE)
        base_satisfaction = rng.uniform(*BASELINE_SATISFACTION)

        current = date(signup.year, signup.month, 1)
        end_month = date(end.year, end.month, 1)
        total_months = max(1, (end_month.year - current.year) * 12 + (end_month.month - current.month))
        month_index = 0

        while current <= end_month:
        
            if cust["churned"] and total_months > 1:
                progress = month_index / (total_months - 1)  # 0 -> 1
                decay = 1.0 - 0.8 * progress
            else:
                decay = 1.0

            month_num = current.month
            if month_num in (6, 7, 8):
                season = 0.8
            elif month_num == 1:
                season = 1.15
            else:
                season = 1.0

            multiplier = decay * season

            logins = max(0, int(base_logins * multiplier + rng.normal(0, 3)))
            sessions = max(0, int(base_sessions * multiplier + rng.normal(0, 2)))
            duration = max(0.5, round(base_duration * multiplier + rng.normal(0, 2), 1))
            feature_usage = max(0.0, min(100.0, round(base_feature * multiplier + rng.normal(0, 5), 1)))
            satisfaction = max(1.0, min(10.0, round(base_satisfaction * multiplier + rng.normal(0, 0.5), 1)))

            if cust["churned"]:
                ticket_rate = 0.3 + 2.0 * (1 - decay) 
            else:
                ticket_rate = 0.3
            support_tickets = int(rng.poisson(ticket_rate))

            if cust["churned"] and decay < 0.5:
                pay_weights = [0.5, 0.3, 0.2]
            elif cust["churned"]:
                pay_weights = [0.7, 0.2, 0.1]
            else:
                pay_weights = [0.92, 0.06, 0.02]
            payment_status = rng.choice(PAYMENT_STATUSES, p=pay_weights)

            records.append({
                "customer_id": cust["customer_id"],
                "month": current.strftime("%Y-%m"),
                "logins": logins,
                "feature_usage_score": feature_usage,
                "total_sessions": sessions,
                "avg_session_duration": duration,
                "support_tickets": support_tickets,
                "payment_status": payment_status,
                "satisfaction_score": satisfaction,
            })

            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
            month_index += 1

    return pd.DataFrame(records)
