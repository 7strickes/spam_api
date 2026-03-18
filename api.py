import pickle

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load artifacts
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
selected_features = pickle.load(open("selected_features.pkl", "rb"))

app = FastAPI(title="Bankruptcy Prediction API")


# ✅ Define input schema manually (MUST match sanitized names)
class InputData(BaseModel):
    Net_Value_Per_Share_B: float
    Net_Value_Per_Share_C: float
    Persistent_EPS_in_the_Last_Four_Seasons: float
    Operating_Profit_Per_Share_Yuan_Yuan: float
    Debt_ratio_percent: float
    Net_worth_Assets: float
    Borrowing_dependency: float
    Current_Liability_to_Assets: float
    Current_Liability_to_Liability: float
    Liability_to_Equity: float


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input → array (correct order)
        input_array = np.array([[getattr(data, feat) for feat in selected_features]])

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        return {
            "prediction": "Bankrupt" if pred == 1 else "Not Bankrupt",
            "probability": float(prob),
        }

    except Exception as e:
        return {"error": str(e)}
