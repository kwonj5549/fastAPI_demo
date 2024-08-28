from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
import pandas as pd
import neurokit2 as nk
import numpy as np
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class ECGInputData(BaseModel):
    ecg_data: list[float]

class ECGOutputData(BaseModel):
    results: dict
    cardiac_score: float

# Store accumulated data in a global variable
accumulated_ecg_data = []

# Minimum chunk size to process data
MIN_CHUNK_SIZE = 500  # Adjust this based on your needs

@app.post("/process_ecg_file", response_model=ECGOutputData)
def process_ecg(
        data: ECGInputData,
        tachycardia_cutoff: int = Query(100, description="Heart rate cutoff for tachycardia"),
        bradycardia_cutoff: int = Query(60, description="Heart rate cutoff for bradycardia")
) -> ECGOutputData:
    try:
        ecg_signal = pd.Series(data.ecg_data)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250)

        # Process ECG and detect R-peaks
        processed_data, info = nk.ecg_process(ecg_signal, sampling_rate=250)
        rpeaks = info['ECG_R_Peaks']

        # Calculate heart rate
        heart_rate = nk.ecg_rate(rpeaks, sampling_rate=250)

        # Detect abnormalities
        tachycardia_periods = np.where(heart_rate > tachycardia_cutoff)[0]
        bradycardia_periods = np.where(heart_rate < bradycardia_cutoff)[0]
        flatline_periods = np.where(np.diff(rpeaks) > 250 * 1.5)[0]  # No R-peaks for more than 1.5 seconds

        # Convert periods to start and end indices
        def get_periods(indices):
            return [[rpeaks[i] / 250, rpeaks[i+1] / 250] for i in indices]

        # Calculate cardiac score
        total_periods = len(heart_rate)
        abnormal_periods = len(tachycardia_periods) + len(bradycardia_periods) + len(flatline_periods)
        cardiac_score = 1 - (abnormal_periods / total_periods)

        # Prepare results
        results = {
            "tachycardia_periods": get_periods(tachycardia_periods),
            "bradycardia_periods": get_periods(bradycardia_periods),
            "flatline_periods": get_periods(flatline_periods),
        }

        return ECGOutputData(results=results, cardiac_score=cardiac_score)
    except Exception as e:
        logging.error(f"Error processing ECG file: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred during processing: {e}")

@app.post("/process_ecg_stream", response_model=ECGOutputData)
async def process_ecg_stream(
        request: Request,
        tachycardia_cutoff: int = Query(100, description="Heart rate cutoff for tachycardia"),
        bradycardia_cutoff: int = Query(60, description="Heart rate cutoff for bradycardia")
) -> ECGOutputData:
    global accumulated_ecg_data
    try:
        body = await request.json()
        if "end_of_stream" in body and body["end_of_stream"]:
            if not accumulated_ecg_data:
                return ECGOutputData(results={"message": "No data to process."}, cardiac_score=1.0)

            # Process accumulated data
            ecg_signal = pd.Series(accumulated_ecg_data)
            ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250)

            # Process ECG and detect R-peaks
            processed_data, info = nk.ecg_process(ecg_signal, sampling_rate=250)
            rpeaks = info['ECG_R_Peaks']

            # Calculate heart rate
            heart_rate = nk.ecg_rate(rpeaks, sampling_rate=250)

            # Detect abnormalities
            tachycardia_periods = np.where(heart_rate > tachycardia_cutoff)[0]
            bradycardia_periods = np.where(heart_rate < bradycardia_cutoff)[0]
            flatline_periods = np.where(np.diff(rpeaks) > 250 * 1.5)[0]  # No R-peaks for more than 1.5 seconds

            # Convert periods to start and end indices
            def get_periods(indices):
                return [[rpeaks[i] / 250, rpeaks[i+1] / 250] for i in indices]

            # Calculate cardiac score
            total_periods = len(heart_rate)
            abnormal_periods = len(tachycardia_periods) + len(bradycardia_periods) + len(flatline_periods)
            cardiac_score = 1 - (abnormal_periods / total_periods)

            # Prepare results
            results = {
                "tachycardia_periods": get_periods(tachycardia_periods),
                "bradycardia_periods": get_periods(bradycardia_periods),
                "flatline_periods": get_periods(flatline_periods),
            }

            # Clear accumulated data after processing
            accumulated_ecg_data = []

            return ECGOutputData(results=results, cardiac_score=cardiac_score)
        else:
            # Accumulate the chunk of data
            chunk_ecg_data = body.get("ecg_data", [])
            if not isinstance(chunk_ecg_data, list):
                raise HTTPException(status_code=400, detail="Invalid data format. Expected a list of floats.")

            accumulated_ecg_data.extend(chunk_ecg_data)

            return ECGOutputData(results={"message": "Chunk received, accumulating data."}, cardiac_score=1.0)

    except Exception as e:
        logging.error(f"Error processing ECG stream: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred during streaming processing: {e}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
