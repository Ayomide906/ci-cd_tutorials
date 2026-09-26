import pandas as pd
from pathlib import Path
import argparse

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def update_training_data():
    print("\n🚀 Running basic feature engineering for Serie A...")
    
    # 1. Load the pre-baked raw CSV instead of a live SQLite database
    raw_path = DATA_DIR / "df1_raw_serieA.csv"
    if not raw_path.exists():
        print(f"❌ Could not find {raw_path}. Make sure the dummy data is in the data folder!")
        return

    df = pd.read_csv(raw_path)

    # 2. The "Dummy" Feature Engineering
    print("Calculating simple match features...")
    df['HomeGoalDiff'] = df['FullTimeHomeGoals'] - df['FullTimeAwayGoals']
    
    # 3. Select a tiny subset of columns for the toy model
    feature_cols = [
        'HomeTeam', 'AwayTeam', 
        'HomeOdds', 'DrawOdds', 'AwayOdds'
    ]
    target_cols = ['FullTimeResult']

    # Drop any rows with missing data to prevent CatBoost crashes
    df = df.dropna(subset=feature_cols + target_cols)

    X_train = df[feature_cols]
    y_train = df[target_cols]

    # 4. Save to League-Specific Files
    out_x = DATA_DIR / "X_train.csv"
    out_y = DATA_DIR / "y_train.csv"
    
    print(f"Saving updated datasets to {DATA_DIR.name}/...")
    X_train.to_csv(out_x, index=False)
    y_train.to_csv(out_y, index=False)
    
    print(f"✅ Success! X_train shape: {X_train.shape}. Ready for MLflow training.")

if __name__ == "__main__":
    update_training_data()