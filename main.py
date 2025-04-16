from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("weather_model.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to Weather Prediction API"}

@app.get("/predict/")
def predict(humidity: float, wind_speed: float):
    prediction = model.predict(np.array([[humidity, wind_speed]]))
    return {"predicted_temperature": round(prediction[0], 2)}
