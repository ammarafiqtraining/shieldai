"""
SMS Spam Preprocessing
Cleans and prepares SMS dataset for training
"""

import pandas as pd
import re
import os

RAW_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\raw\sms_spam.csv"
OUTPUT_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\processed\sms_cleaned.csv"

def load_raw_sms(path: str) -> pd.DataFrame:
    """Load and clean raw SMS dataset."""
    print("[LOADING RAW DATA]")
    
    df = pd.read_csv(path, header=None, names=['label', 'message', 'extra1', 'extra2', 'extra3'], encoding='latin-1')
    
    df = df[['label', 'message']].copy()
    
    df = df.dropna(subset=['label', 'message'])
    
    df = df[df['label'].isin(['ham', 'spam'])]
    
    print(f"  Loaded: {len(df)} messages")
    print(f"  Ham: {(df['label'] == 'ham').sum()}")
    print(f"  Spam: {(df['label'] == 'spam').sum()}")
    
    return df


def clean_message(text: str) -> str:
    """Clean and normalize SMS text."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    return text


def preprocess_sms(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess all SMS messages."""
    print("\n[CLEANING MESSAGES]")
    
    df['message_clean'] = df['message'].apply(clean_message)
    
    df = df[df['message_clean'].str.len() > 0]
    
    df['ml_label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    original = len(df)
    df = df.drop_duplicates(subset=['message_clean'])
    print(f"  Removed {original - len(df)} duplicates")
    
    print(f"  Final: {len(df)} messages")
    
    return df


def save_processed(df: pd.DataFrame, path: str):
    """Save processed dataset."""
    print(f"\n[SAVING] {path}")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df[['message_clean', 'ml_label', 'message']].to_csv(path, index=False)
    
    print(f"  Saved: {len(df)} messages")


def main():
    print("=" * 50)
    print("SMS SPAM PREPROCESSING")
    print("=" * 50)
    
    df = load_raw_sms(RAW_PATH)
    
    df = preprocess_sms(df)
    
    save_processed(df, OUTPUT_PATH)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return df


if __name__ == "__main__":
    main()