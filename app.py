# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle, json, os

# --- Load model + features at startup ---
MODEL_DIR = os.getenv("MODEL_DIR", "model")
DATA_DIR = os.getenv("DATA_DIR", "data")

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "model_features.json"), "r") as f:
    model_features = json.load(f)

demo_df = pd.read_csv(os.path.join(DATA_DIR, "zipcode_demographics.csv"),
                      dtype={"zipcode": str})

# --- Request schemas ---
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class HouseFeaturesMinimal(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: str

app = FastAPI(title="Sound Realty Price API", version="1.0")

def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(demo_df, on="zipcode", how="left")
    df = df.drop(columns=["zipcode"])
    # align to the exact training order
    missing = [c for c in model_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return df[model_features]

@app.post("/predict")
def predict_full(data: HouseFeatures):
    row = pd.DataFrame([data.dict()])
    X = prepare_input(row)
    yhat = model.predict(X)[0]
    return {"predicted_price": float(yhat), "model_version": "1.0", "unit": "USD"}

@app.post("/predict_minimal")
def predict_minimal(data: HouseFeaturesMinimal):
    row = pd.DataFrame([data.dict()])
    X = prepare_input(row)
    yhat = model.predict(X)[0]
    return {"predicted_price": float(yhat), "model_version": "1.0", "unit": "USD"}

