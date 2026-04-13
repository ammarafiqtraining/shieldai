import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import config


def train_random_forest(X_train, y_train):
    print("\n[TRAINING RANDOM FOREST]")
    print(f"  Parameters: {config.RF_PARAMS}")
    
    model = RandomForestClassifier(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    
    return model


def train_logistic_regression(X_train, y_train):
    print("\n[TRAINING LOGISTIC REGRESSION]")
    print(f"  Parameters: {config.LR_PARAMS}")
    
    model = LogisticRegression(**config.LR_PARAMS)
    model.fit(X_train, y_train)
    
    return model


def save_models(models_dict, feature_names, metadata):
    print("\n[SAVING MODELS]")
    
    model_versions = {}
    
    for model_type, model in models_dict.items():
        model_path = os.path.join(
            config.MODELS_DIR,
            config.get_model_filename(model_type, config.PROCESSED_VERSION)
        )
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")
        model_versions[model_type] = config.get_model_filename(model_type, config.PROCESSED_VERSION)
    
    feature_names_path = os.path.join(config.MODELS_DIR, "feature_names.pkl")
    joblib.dump(feature_names, feature_names_path)
    print(f"  Saved: {feature_names_path}")
    
    feature_importance = None
    if "rf" in models_dict:
        rf_model = models_dict["rf"]
        importance = rf_model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        top_features = [
            {"feature": feature_names[i], "importance": float(importance[i])}
            for i in sorted_idx[:20]
        ]
        print(f"\n  Top 20 Important Features (RF):")
        for i, feat in enumerate(top_features, 1):
            print(f"    {i:2d}. {feat['feature']}: {feat['importance']:.4f}")

    model_info = {
        "version": config.PROCESSED_VERSION,
        "created_at": config.get_timestamp(),
        "models": model_versions,
        "feature_names_file": "feature_names.pkl",
        "data_version": metadata["version"],
        "target_metric": config.TARGET_METRIC,
        "target_threshold": config.TARGET_THRESHOLD,
        "rf_params": config.RF_PARAMS,
        "lr_params": config.LR_PARAMS,
        "feature_importance_top20": top_features if feature_importance is None else None
    }
    
    model_info_path = os.path.join(config.MODELS_DIR, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\n  Saved: {model_info_path}")


def main(X_train, y_train, feature_names, metadata):
    print("=" * 60)
    print("PHISHING DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    models = {}
    
    models["rf"] = train_random_forest(X_train, y_train)
    models["lr"] = train_logistic_regression(X_train, y_train)
    
    save_models(models, feature_names, metadata)
    
    print("\n[TRAINING COMPLETE]")
    return models


if __name__ == "__main__":
    from preprocess import main as preprocess_main
    
    X_train, X_test, y_train, y_test, feature_names, metadata = preprocess_main()
    main(X_train, y_train, feature_names, metadata)
