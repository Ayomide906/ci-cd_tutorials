import pandas as pd
from pathlib import Path

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def fetch_and_ingest():
    print("🚀 Starting Data Ingestion for Serie A...")
    
    # 1. Fetch data directly from football-data.co.uk 
    # Note: 'I1.csv' is the official code for Italy Serie A. 
    # We use a stable, completed season (2324) so the bootcamp pipeline never fails mid-class.
    url = "https://www.football-data.co.uk/mmz4281/2627/I1.csv"
    print(f"Downloading raw match data from {url}...")
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return

    # 2. Filter and Rename Columns for the Dummy Pipeline
    # We strictly grab what build_features.py needs, ignoring the 100+ other stat columns.
    columns_to_keep = {
        'HomeTeam': 'HomeTeam',
        'AwayTeam': 'AwayTeam',
        'FTHG': 'FullTimeHomeGoals',
        'FTAG': 'FullTimeAwayGoals',
        'FTR': 'FullTimeResult',
        'B365H': 'HomeOdds',
        'B365D': 'DrawOdds',
        'B365A': 'AwayOdds'
    }
    
    # Apply the filter and rename
    df = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
    
    # Drop any rows with missing odds or goals to prevent downstream crashes
    df = df.dropna()

    # 3. Save to a flat CSV file instead of SQLite
    out_path = DATA_DIR / "serie_a_raw.csv"
    df.to_csv(out_path, index=False)
    
    print(f"✅ Successfully ingested {len(df)} matches!")
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    fetch_and_ingest()