"""Configuration and schema definitions for data generation."""

NUM_CUSTOMERS = 2000
RANDOM_SEED = 42

PLAN_TYPES = ["Basic", "Standard", "Premium"]
PLAN_PRICES = {
    "Basic": (9.99, 14.99),
    "Standard": (19.99, 29.99),
    "Premium": (39.99, 59.99),
}

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]

ACQ_CHANNELS = ["referral", "ad", "organic", "social media"]
ACQ_WEIGHTS = [0.25, 0.30, 0.25, 0.20]

CONTRACT_TYPES = ["Month-to-month", "Annual"]
CONTRACT_WEIGHTS = [0.65, 0.35]

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Seattle", "Denver", "Boston", "Nashville",
    "Portland", "Atlanta", "Miami", "Minneapolis", "Detroit",
]

PAYMENT_STATUSES = ["On-time", "Late", "Failed"]

CHURN_RATE = 0.25

AGE_MIN = 18
AGE_MAX = 72
AGE_MEAN = 35
AGE_STD = 10

BASELINE_LOGINS = (15, 30)
BASELINE_SESSIONS = (10, 25)
BASELINE_SESSION_DURATION = (8.0, 30.0)
BASELINE_FEATURE_USAGE = (50.0, 90.0)
BASELINE_SATISFACTION = (6.0, 9.0)

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

TEST_SIZE = 0.2