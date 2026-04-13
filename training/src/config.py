import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = r"training/data/raw/dataset_phishing.csv"

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FEEDBACK_DATA_DIR = os.path.join(BASE_DIR, "data", "feedback")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PROCESSED_VERSION = "v1"

TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

LABEL_MAPPING = {
    "legitimate": 0,
    "phishing": 1
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}

LR_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs"
}

TARGET_METRIC = "recall"
TARGET_THRESHOLD = 0.90

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

def get_metadata_filename(version):
    return f"metadata_{version}.json"

def get_processed_filename(version):
    return f"dataset_{version}.csv"

def get_model_filename(model_type, version):
    return f"phishing_model_{model_type}_{version}.pkl"
