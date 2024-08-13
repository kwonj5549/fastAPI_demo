from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import pandas as pd
import neurokit2 as nk
import numpy as np

app = FastAPI()

class ECGInputData(BaseModel):
    ecg_data: conlist(float)  # Ensure at least one data point is provided

class ECGOutputData(BaseModel):
    results: dict

@app.post("/process_ecg", response_model=ECGOutputData)
def process_ecg(data: ECGInputData) -> ECGOutputData:
    try:
        # Create a DataFrame with the provided ECG data
        ecg_signal = pd.Series(data.ecg_data)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=1000)  # Clean the signal

        # Preprocess the data (filter, find peaks, etc.)
        processed_data, info = nk.bio_process(ecg=ecg_signal, sampling_rate=1000)

        # Compute relevant features
        results = nk.bio_analyze(processed_data, sampling_rate=1000)

        # Convert results to a dictionary and return it as a response
        return ECGOutputData(results=results.to_dict(orient='records'))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred during processing: {e}")

# Basic root endpoint for testing
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Example endpoint for demonstration purposes
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
