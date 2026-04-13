import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config


class DataValidator:
    def __init__(self, df):
        self.df = df
        self.issues = []
        self.warnings = []

    def check_duplicates(self):
        duplicates = self.df.duplicated(subset=["url"], keep=False)
        if duplicates.any():
            self.warnings.append(f"Found {duplicates.sum()} duplicate URLs")
            self.df = self.df.drop_duplicates(subset=["url"], keep="first")
            print(f"  [WARNING] Removed {duplicates.sum()} duplicate URLs")
        return self

    def check_missing_values(self):
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            self.issues.append(f"Missing values in columns: {missing_cols.to_dict()}")
            self.df = self.df.dropna()
            print(f"  [ERROR] Dropped {len(missing_cols)} rows with missing values")
        return self

    def check_class_balance(self):
        label_counts = self.df["status"].value_counts()
        min_count = label_counts.min()
        max_count = label_counts.max()
        balance_ratio = min_count / max_count
        
        if balance_ratio < 0.5:
            self.warnings.append(
                f"Imbalanced dataset: ratio={balance_ratio:.2f}. "
                f"Legitimate: {label_counts.get('legitimate', 0)}, "
                f"Phishing: {label_counts.get('phishing', 0)}"
            )
            print(f"  [WARNING] Imbalanced dataset detected (ratio: {balance_ratio:.2f})")
        else:
            print(f"  [OK] Class balance acceptable (ratio: {balance_ratio:.2f})")
        return self

    def check_data_leakage(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "status" in numeric_cols:
            numeric_cols.remove("status")
        
        suspicious_cols = [col for col in numeric_cols if any(
            keyword in col.lower() for keyword in ["future", "prediction", "confidence"]
        )]
        if suspicious_cols:
            self.issues.append(f"Potential leakage columns: {suspicious_cols}")
            print(f"  [WARNING] Suspicious columns found: {suspicious_cols}")
        return self

    def validate(self):
        print("\n[DATA VALIDATION]")
        self.check_duplicates()
        self.check_missing_values()
        self.check_class_balance()
        self.check_data_leakage()
        
        if self.issues:
            print(f"\n  [CRITICAL ISSUES] {len(self.issues)}")
            for issue in self.issues:
                print(f"    - {issue}")
        if self.warnings:
            print(f"\n  [WARNINGS] {len(self.warnings)}")
            for warning in self.warnings:
                print(f"    - {warning}")
        
        return len(self.issues) == 0


class DataPreprocessor:
    def __init__(self, raw_path, version):
        self.raw_path = raw_path
        self.version = version
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        print("\n[LOADING DATA]")
        print(f"  Source: {self.raw_path}")
        self.df = pd.read_csv(self.raw_path)
        print(f"  Loaded {len(self.df)} samples with {len(self.df.columns)} columns")
        return self

    def preprocess(self):
        print("\n[PREPROCESSING]")
        
        feature_cols = [col for col in self.df.columns if col not in ["url", "status"]]
        self.feature_names = feature_cols
        
        self.X = self.df[feature_cols].copy()
        self.y = self.df["status"].map(config.LABEL_MAPPING)
        
        print(f"  Features: {len(feature_cols)}")
        print(f"  Label distribution: {dict(self.y.value_counts())}")
        return self

    def split_data(self):
        print("\n[SPLITTING DATA]")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=self.y if config.STRATIFY else None
        )
        print(f"  Train size: {len(self.X_train)}")
        print(f"  Test size: {len(self.X_test)}")
        return self

    def save_processed_data(self):
        print("\n[SAVING PROCESSED DATA]")
        
        processed_df = self.X.copy()
        processed_df["label"] = self.y
        processed_df["url"] = self.df["url"]
        
        processed_path = os.path.join(
            config.PROCESSED_DATA_DIR,
            config.get_processed_filename(self.version)
        )
        processed_df.to_csv(processed_path, index=False)
        print(f"  Saved: {processed_path}")

        label_counts = self.y.value_counts().to_dict()
        metadata = {
            "version": self.version,
            "source": self.raw_path,
            "created_at": config.get_timestamp(),
            "total_samples": int(len(self.df)),
            "legitimate_count": int(label_counts.get(0, 0)),
            "phishing_count": int(label_counts.get(1, 0)),
            "features_used": len(self.feature_names),
            "feature_names": self.feature_names,
            "preprocessing_steps": [
                "duplicate_removal",
                "missing_value_handling",
                "label_encoding",
                "train_test_split"
            ],
            "train_samples": int(len(self.X_train)),
            "test_samples": int(len(self.X_test)),
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE
        }

        metadata_path = os.path.join(
            config.PROCESSED_DATA_DIR,
            config.get_metadata_filename(self.version)
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {metadata_path}")
        
        return metadata

    def run(self):
        self.load_data()
        
        validator = DataValidator(self.df)
        is_valid = validator.validate()
        
        if not is_valid:
            raise ValueError("Data validation failed. Please fix critical issues before proceeding.")
        
        self.df = validator.df
        
        self.preprocess()
        self.split_data()
        metadata = self.save_processed_data()
        
        print("\n[PREPROCESSING COMPLETE]")
        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names, metadata


def main():
    print("=" * 60)
    print("PHISHING DETECTION - DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessor = DataPreprocessor(config.DATA_PATH, config.PROCESSED_VERSION)
    X_train, X_test, y_train, y_test, feature_names, metadata = preprocessor.run()
    
    return X_train, X_test, y_train, y_test, feature_names, metadata


if __name__ == "__main__":
    main()
