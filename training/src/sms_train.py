"""
SMS Spam Model Training
Uses TF-IDF + Logistic Regression
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

DATA_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\processed\sms_cleaned.csv"
MODEL_DIR = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\models\sms"


def load_data(path: str) -> pd.DataFrame:
    """Load processed SMS data."""
    print("[LOADING DATA]")
    df = pd.read_csv(path)
    print(f"  Total: {len(df)} messages")
    print(f"  Ham: {(df['ml_label'] == 0).sum()}, Spam: {(df['ml_label'] == 1).sum()}")
    return df


def prepare_features(df: pd.DataFrame):
    """Create TF-IDF features."""
    print("\n[CREATING TF-IDF FEATURES]")
    
    X = df['message_clean']
    y = df['ml_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Train: {X_train_tfidf.shape}")
    print(f"  Test: {X_test_tfidf.shape}")
    print(f"  Vocabulary: {len(vectorizer.vocabulary_)} terms")
    
    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test


def train_model(X_train, y_train) -> LogisticRegression:
    """Train Logistic Regression model."""
    print("\n[TRAINING MODEL]")
    
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("  Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, vectorizer) -> dict:
    """Evaluate and return metrics."""
    print("\n[EVALUATING MODEL]")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    print("\n  Metrics:")
    for name, value in metrics.items():
        status = "[PASS]" if value >= 0.85 else "[FAIL]"
        print(f"    {name.capitalize():12s}: {value:.4f} {status}")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    
    top_features = get_top_features(model, vectorizer)
    print("\n  Top 10 Spam Indicators:")
    for word, weight in top_features:
        print(f"    {word}: {weight:.4f}")
    
    return metrics


def get_top_features(model, vectorizer):
    """Get top features indicating spam."""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    top_indices = np.argsort(coefficients)[-10:][::-1]
    
    return [(feature_names[i], coefficients[i]) for i in top_indices]


def save_artifacts(vectorizer, model, metrics):
    """Save model and metadata."""
    print("\n[SAVING ARTIFACTS]")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    vectorizer_path = os.path.join(MODEL_DIR, "sms_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  Vectorizer: {vectorizer_path}")
    
    model_path = os.path.join(MODEL_DIR, "sms_model.pkl")
    joblib.dump(model, model_path)
    print(f"  Model: {model_path}")
    
    metadata = {
        "version": "v1",
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type": "LogisticRegression",
        "vectorizer_type": "TF-IDF",
        "max_features": 5000,
        "ngram_range": [1, 2],
        "training_samples": 4101,
        "test_samples": 1026,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "note": "SMS spam detection model using TF-IDF"
    }
    
    metadata_path = os.path.join(MODEL_DIR, "sms_model_info.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    
    return model_path, metadata_path


def main():
    print("=" * 50)
    print("SMS SPAM MODEL TRAINING")
    print("=" * 50)
    
    df = load_data(DATA_PATH)
    
    vectorizer, X_train, X_test, y_train, y_test = prepare_features(df)
    
    model = train_model(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, vectorizer)
    
    save_artifacts(vectorizer, model, metrics)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1: {metrics['f1']:.2%}")
    
    return model, vectorizer


if __name__ == "__main__":
    main()