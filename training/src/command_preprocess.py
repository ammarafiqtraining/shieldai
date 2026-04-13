"""
Command Preprocessing
Cleans and prepares command dataset for training
"""

import pandas as pd
import re
import os

RAW_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\raw\command_synthetic.csv"
OUTPUT_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\processed\command_cleaned.csv"

MAX_LENGTH = 500


def load_raw_commands(path: str) -> pd.DataFrame:
    """Load raw command dataset."""
    print("[LOADING RAW DATA]")
    
    df = pd.read_csv(path, encoding='utf-8')
    
    print(f"  Loaded: {len(df)} commands")
    print(f"  Malicious: {(df['label'] == 1).sum()}")
    print(f"  Legitimate: {(df['label'] == 0).sum()}")
    
    return df


def clean_command(text: str) -> str:
    """Clean and normalize command text."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s\.\-\/\\]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    
    return text


def preprocess_commands(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess all commands."""
    print("\n[CLEANING COMMANDS]")
    
    df['command_clean'] = df['command'].apply(clean_command)
    
    df = df[df['command_clean'].str.len() > 0]
    print(f"  After empty filter: {len(df)} commands")
    
    df = df.drop_duplicates(subset=['command_clean'])
    print(f"  After dedup: {len(df)} commands")
    
    df['ml_label'] = df['label'].astype(int)
    
    print(f"  Final: {len(df)} commands")
    
    return df


def save_processed(df: pd.DataFrame, path: str):
    """Save processed dataset."""
    print(f"\n[SAVING] {path}")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df[['command_clean', 'ml_label', 'command', 'category']].to_csv(path, index=False)
    
    print(f"  Saved: {len(df)} commands")


def main():
    print("=" * 50)
    print("COMMAND PREPROCESSING")
    print("=" * 50)
    
    df = load_raw_commands(RAW_PATH)
    
    df = preprocess_commands(df)
    
    save_processed(df, OUTPUT_PATH)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return df


if __name__ == "__main__":
    main()