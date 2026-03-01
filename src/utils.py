"""
Utility functions for saving/loading data and models.
"""

import os

import joblib
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def save_csv(df: pd.DataFrame, filename: str) -> str:

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path


def load_csv(filename: str) -> pd.DataFrame:

    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


def save_model(model, filename: str) -> str:
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(filename: str):

    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)