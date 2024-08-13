from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import neurokit2 as nk
import numpy as np
import json
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class ECGInputData(BaseModel):
    ecg_data: list[float]

class ECGOutputData(BaseModel):
    results: dict

# Store accumulated data in a global variable
accumulated_ecg_data = []

# Minimum chunk size to process data
MIN_CHUNK_SIZE = 500  # Adjust this based on your needs

@app.post("/process_ecg_file", response_model=ECGOutputData)
def process_ecg(data: ECGInputData) -> ECGOutputData:
    try:
        ecg_signal = pd.Series(data.ecg_data)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=1000)

        processed_data, info = nk.bio_process(ecg=ecg_signal, sampling_rate=1000)
        results = nk.bio_analyze(processed_data, sampling_rate=1000)

        results_dict = results.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='list')
        return ECGOutputData(results=results_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred during processing: {e}")

@app.post("/process_ecg_stream", response_model=ECGOutputData)
async def process_ecg_stream(request: Request) -> ECGOutputData:
    global accumulated_ecg_data
    try:
        body = await request.json()
        if "end_of_stream" in body and body["end_of_stream"]:
            if not accumulated_ecg_data:
                return {"message": "No data to process."}

            # Process accumulated data
            ecg_signal = pd.Series(accumulated_ecg_data)
            ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=1000)

            processed_data, info = nk.bio_process(ecg=ecg_signal, sampling_rate=1000)
            results = nk.bio_analyze(processed_data, sampling_rate=1000)

            results_dict = results.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='list')

            # Clear accumulated data after processing
            accumulated_ecg_data = []

            return ECGOutputData(results=results_dict)
        else:
            # Accumulate the chunk of data
            chunk_ecg_data = body.get("ecg_data", [])
            if not isinstance(chunk_ecg_data, list):
                raise HTTPException(status_code=400, detail="Invalid data format. Expected a list of floats.")

            accumulated_ecg_data.extend(chunk_ecg_data)

            return {"message": "Chunk received, accumulating data."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred during streaming processing: {e}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
