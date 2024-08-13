from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import neurokit2 as nk
import numpy as np

app = FastAPI()

class ECGInputData(BaseModel):
    ecg_data: list[float]  # A list of floating-point numbers representing the ECG data

class ECGOutputData(BaseModel):
    results: dict  # Expecting a dictionary of results

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

        # Convert the DataFrame results to a dictionary
        results_dict = results.to_dict(orient='list')

        # Return the dictionary in the correct format
        return ECGOutputData(results=results_dict)

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
