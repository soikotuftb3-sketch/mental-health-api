import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("mental_health_rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

class InputData(BaseModel):
    age: int
    gender: str
    occupation: str
    ...
