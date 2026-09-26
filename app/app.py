import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "SerieAresult_model.pkl"

# 2. Initialize API
app = FastAPI(title="Serie A Predictor (CI/CD Bootcamp)", version="1.0")

# 3. Load the Dummy Model globally at startup
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("⚠️ Warning: Model not found. Did you run train.py to generate the .pkl?")

# 4. Define the Schema (Must match the 6 dummy features)
class MatchRequest(BaseModel):
    HomeTeam: str
    AwayTeam: str
    HomeOdds: float
    DrawOdds: float
    AwayOdds: float
# 5. Health Check Endpoint (For Docker & GitHub Actions to test)
@app.get("/")
def health_check():
    return {"status": "online", "message": "Bootcamp API is ready for predictions."}

# 6. Single Prediction Endpoint
@app.post("/predict")
def predict_match(match: MatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Convert the incoming JSON payload into a Pandas DataFrame
        input_data = pd.DataFrame([match.model_dump()])
        
        # Run prediction
        raw_pred = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        
        # Format the probabilities neatly
        prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        
        return {
            "match": f"{match.HomeTeam} vs {match.AwayTeam}",
            "predicted_outcome": str(raw_pred),
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))