import pandas as pd

class LiveMatchFeatureEngineer:
    def __init__(self, df1_raw=None, team_df_raw=None):
        """
        Simplified for the CI/CD Bootcamp.
        Historical datasets are ignored since we only need 5 basic features.
        """
        # Target feature list exact order
        self.feature_columns = [
            'HomeTeam', 'AwayTeam', 'HomeOdds', 'DrawOdds', 'AwayOdds'
        ]

    def fit(self, X=None, y=None):
        """
        Dummy fit method. 
        Keeps app.py from breaking when it calls PIPELINES[league].fit()
        """
        return self

    def transform(self, home_team, away_team, season, home_odds, draw_odds, away_odds):
        """
        Returns only the 5 required features for the dummy model.
        """
        row = {
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "HomeOdds": home_odds,
            "DrawOdds": draw_odds,
            "AwayOdds": away_odds
        }

        # Create DataFrame and guarantee exact column order for CatBoost
        result_df = pd.DataFrame([row])
        return result_df[self.feature_columns]

# --- Usage Example ---
if __name__ == '__main__':
    # You can pass None since we don't need the historical CSVs for this simple model
    pipeline = LiveMatchFeatureEngineer()
    pipeline.fit()
    
    sample_features = pipeline.transform(
        home_team="Torino",
        away_team="Cagliari",
        season="2025/2026",
        home_odds=1.85,
        draw_odds=3.60,
        away_odds=4.50
    )
    
    print(f"Feature Vector Shape: {sample_features.shape}") # Should be (1, 5)
    print("\nFeatures:")
    print(sample_features)