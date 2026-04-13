"""
Email Phishing Preprocessing
Cleans and prepares CEAS_08 email dataset for training
"""

import pandas as pd
import re
import os

RAW_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\raw\CEAS_08.csv"
OUTPUT_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\processed\email_cleaned.csv"

MAX_BODY_LENGTH = 10000


def load_raw_email(path: str) -> pd.DataFrame:
    """Load and validate CEAS_08 dataset."""
    print("[LOADING RAW DATA]")
    
    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    
    print(f"  Loaded: {len(df)} emails")
    print(f"  Columns: {df.columns.tolist()}")
    
    df = df[['subject', 'body', 'label', 'sender']].copy()
    
    df = df.dropna(subset=['subject', 'body', 'label'])
    
    print(f"  After dropna: {len(df)} emails")
    print(f"  Phishing (1): {(df['label'] == 1).sum()}")
    print(f"  Legitimate (0): {(df['label'] == 0).sum()}")
    
    return df


def clean_text(text: str) -> str:
    """Clean and normalize email text."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    text = text.lower()
    
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = re.sub(r'http[s]?://\S+', ' url ', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    return text


def clean_body(text: str) -> str:
    """Clean body with length limit."""
    cleaned = clean_text(text)
    if len(cleaned) > MAX_BODY_LENGTH:
        cleaned = cleaned[:MAX_BODY_LENGTH]
    return cleaned


def preprocess_email(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess all email data."""
    print("\n[CLEANING DATA]")
    
    df['subject_clean'] = df['subject'].apply(clean_text)
    df['body_clean'] = df['body'].apply(clean_body)
    
    df['text_combined'] = df['subject_clean'] + ' ' + df['body_clean']
    
    df = df[df['text_combined'].str.len() > 10]
    print(f"  After empty filter: {len(df)} emails")
    
    original = len(df)
    df = df.drop_duplicates(subset=['text_combined'])
    print(f"  Removed {original - len(df)} duplicates")
    
    df['ml_label'] = df['label'].astype(int)
    
    print(f"  Final: {len(df)} emails")
    
    return df


def save_processed(df: pd.DataFrame, path: str):
    """Save processed dataset."""
    print(f"\n[SAVING] {path}")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df[['text_combined', 'subject_clean', 'body_clean', 'ml_label', 'sender', 'subject', 'body']].to_csv(path, index=False)
    
    print(f"  Saved: {len(df)} emails")


def main():
    print("=" * 50)
    print("EMAIL PHISHING PREPROCESSING (CEAS_08)")
    print("=" * 50)
    
    df = load_raw_email(RAW_PATH)
    
    df = preprocess_email(df)
    
    save_processed(df, OUTPUT_PATH)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return df


if __name__ == "__main__":
    main()