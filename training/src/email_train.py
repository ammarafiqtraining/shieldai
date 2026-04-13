"""
Email Phishing Training
TF-IDF vectorization + Logistic Regression
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATA_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\processed\email_cleaned.csv"
MODEL_DIR = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\models\email"

MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> tuple:
    """Load processed email data."""
    print("[LOADING DATA]")
    
    df = pd.read_csv(path)
    
    X = df['text_combined'].fillna('').values
    y = df['ml_label'].values
    
    print(f"  Total samples: {len(X)}")
    print(f"  Phishing: {(y == 1).sum()}")
    print(f"  Legitimate: {(y == 0).sum()}")
    
    return X, y


def train_model(X_train, y_train, X_test, y_test):
    """Train TF-IDF + Logistic Regression model."""
    print("\n[CREATING TF-IDF VECTORIZER]")
    
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train shape: {X_train_tfidf.shape}")
    print(f"  Test shape: {X_test_tfidf.shape}")
    
    print("\n[TRAINING LOGISTIC REGRESSION]")
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train_tfidf, y_train)
    
    print("  Training complete")
    
    y_pred = model.predict(X_test_tfidf)
    
    print("\n[METRICS]")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:   {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'phishing']))
    
    return model, vectorizer, {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred), 4)
    }


def save_model(model, vectorizer, metrics, total_samples):
    """Save model and vectorizer."""
    print("\n[SAVING MODEL]")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "email_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "email_vectorizer.pkl")
    info_path = os.path.join(MODEL_DIR, "email_model_info.json")
    
    joblib.dump(model, model_path)
    print(f"  Model: {model_path}")
    
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  Vectorizer: {vectorizer_path}")
    
    info = {
        "version": "v1",
        "model_type": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "max_features": MAX_FEATURES,
        "ngram_range": [1, 2],
        "training_samples": int(total_samples * (1 - TEST_SIZE)),
        "test_samples": int(total_samples * TEST_SIZE),
        "metrics": metrics,
        "dataset": "CEAS_08",
        "label_mapping": {"0": "legitimate", "1": "phishing"}
    }
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Info: {info_path}")
    
    return info


def main():
    print("=" * 60)
    print("EMAIL PHISHING MODEL TRAINING")
    print("=" * 60)
    
    X, y = load_data(DATA_PATH)
    total_samples = len(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    model, vectorizer, metrics = train_model(X_train, y_train, X_test, y_test)
    
    info = save_model(model, vectorizer, metrics, total_samples)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return model, vectorizer, info


if __name__ == "__main__":
    main()