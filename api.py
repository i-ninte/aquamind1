from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()

class WaterQualityInput(BaseModel):
    temperature: float
    do: float
    ph: float
    conductivity: float
    bod: float

@app.post("/predict")
async def predict(input_data: WaterQualityInput):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        # Preprocess data
        data.columns = data.columns.str.lower()
        scaled_data = scaler.transform(data)
        # Make prediction
        prediction = model.predict(scaled_data)
        quality = "Potable" if prediction[0] == 1 else "Not Potable"
        return {"quality": quality}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

