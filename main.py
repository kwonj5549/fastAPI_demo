from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
import pandas as pd
import neurokit2 as nk
import numpy as np
import logging
from google.cloud import storage  # GCP Storage client
import os
import io

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

# GCP Bucket name and path
BUCKET_NAME = 'fastapibucket'
STORAGE_CLIENT = storage.Client()

def upload_to_gcs(file_name: str, data: pd.DataFrame, metadata: dict = None):
    try:
        bucket = STORAGE_CLIENT.bucket(BUCKET_NAME)
        blob = bucket.blob(file_name)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(data.to_csv(index=False), 'text/csv')
        logging.info(f"Uploaded {file_name} to GCS with metadata {metadata}")
    except Exception as e:
        logging.error(f"Failed to upload {file_name} to GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading to GCS: {e}")

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Check for overlap
            merged[-1] = [last[0], max(last[1], current[1])]  # Merge intervals
        else:
            merged.append(current)
    return merged

@app.post("/process_ecg_file", response_model=ECGOutputData)
def process_ecg(
        data: ECGInputData,
        tachycardia_cutoff: int = Query(100, description="Heart rate cutoff for tachycardia"),
        bradycardia_cutoff: int = Query(60, description="Heart rate cutoff for bradycardia"),
        timestamp: str = Query(None, description="Optional timestamp in ISO 8601 format")
) -> ECGOutputData:
    try:
        # Parse timestamp
        if timestamp:
            try:
                timestamp_parsed = pd.Timestamp(timestamp)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 format.")
        else:
            timestamp_parsed = pd.Timestamp.now()

        timestamp_str = timestamp_parsed.isoformat()
        timestamp_for_filename = timestamp_parsed.strftime('%Y%m%d_%H%M%S')

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
            return [[rpeaks[i] / 250, rpeaks[i+1] / 250] for i in indices if i+1 < len(rpeaks)]

        # Merge overlapping intervals
        merged_tachycardia = merge_intervals(get_periods(tachycardia_periods.tolist()))
        merged_bradycardia = merge_intervals(get_periods(bradycardia_periods.tolist()))
        merged_flatline = merge_intervals(get_periods(flatline_periods.tolist()))

        # Calculate cardiac score
        total_periods = len(heart_rate)
        abnormal_periods = len(tachycardia_periods) + len(bradycardia_periods) + len(flatline_periods)
        cardiac_score = 1 - (abnormal_periods / total_periods)

        # Prepare results
        results = {
            "tachycardia_periods": merged_tachycardia,
            "bradycardia_periods": merged_bradycardia,
            "flatline_periods": merged_flatline,
        }

        # Upload ECG data to GCS
        df = pd.DataFrame(data.ecg_data, columns=['ECG Data'])
        file_name = f"ecg_data_{timestamp_for_filename}.csv"
        metadata = {'timestamp': timestamp_str}
        upload_to_gcs(file_name, df, metadata=metadata)

        return ECGOutputData(results=results, cardiac_score=cardiac_score)
    except Exception as e:
        logging.error(f"Error processing ECG file: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred during processing: {e}")

@app.post("/process_ecg_stream", response_model=ECGOutputData)
async def process_ecg_stream(
        request: Request,
        tachycardia_cutoff: int = Query(100, description="Heart rate cutoff for tachycardia"),
        bradycardia_cutoff: int = Query(60, description="Heart rate cutoff for bradycardia"),
        timestamp: str = Query(None, description="Optional timestamp in ISO 8601 format")
) -> ECGOutputData:
    global accumulated_ecg_data
    try:
        body = await request.json()
        if "end_of_stream" in body and body["end_of_stream"]:
            if not accumulated_ecg_data:
                return ECGOutputData(results={"message": "No data to process."}, cardiac_score=1.0)

            # Parse timestamp
            if timestamp:
                try:
                    timestamp_parsed = pd.Timestamp(timestamp)
                except Exception as e:
                    raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 format.")
            else:
                timestamp_parsed = pd.Timestamp.now()

            timestamp_str = timestamp_parsed.isoformat()
            timestamp_for_filename = timestamp_parsed.strftime('%Y%m%d_%H%M%S')

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
                return [[rpeaks[i] / 250, rpeaks[i+1] / 250] for i in indices if i+1 < len(rpeaks)]

            # Merge overlapping intervals
            merged_tachycardia = merge_intervals(get_periods(tachycardia_periods.tolist()))
            merged_bradycardia = merge_intervals(get_periods(bradycardia_periods.tolist()))
            merged_flatline = merge_intervals(get_periods(flatline_periods.tolist()))

            # Calculate cardiac score
            total_periods = len(heart_rate)
            abnormal_periods = len(tachycardia_periods) + len(bradycardia_periods) + len(flatline_periods)
            cardiac_score = 1 - (abnormal_periods / total_periods)

            # Prepare results
            results = {
                "tachycardia_periods": merged_tachycardia,
                "bradycardia_periods": merged_bradycardia,
                "flatline_periods": merged_flatline,
            }

            # Upload ECG data to GCS
            df = pd.DataFrame(accumulated_ecg_data, columns=['ECG Data'])
            file_name = f"ecg_stream_data_{timestamp_for_filename}.csv"
            metadata = {'timestamp': timestamp_str}
            upload_to_gcs(file_name, df, metadata=metadata)

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

@app.get("/statistics")
def get_statistics(
        period: str = Query('daily', description="Period for statistics: 'daily' or 'weekly'")
) -> dict:
    try:
        now = pd.Timestamp.now()
        if period == 'daily':
            start_time = now - pd.Timedelta(days=1)
        elif period == 'weekly':
            start_time = now - pd.Timedelta(days=7)
        else:
            raise HTTPException(status_code=400, detail="Invalid period. Choose 'daily' or 'weekly'.")

        bucket = STORAGE_CLIENT.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs()

        # Collect data from blobs within the time range
        ecg_data_list = []

        for blob in blobs:
            # Retrieve the timestamp from metadata
            blob.reload()  # Refresh metadata
            metadata = blob.metadata or {}
            timestamp_str = metadata.get('timestamp')
            if not timestamp_str:
                continue  # Skip if no timestamp metadata

            try:
                file_timestamp = pd.Timestamp(timestamp_str)
            except Exception as e:
                logging.warning(f"Invalid timestamp in metadata for blob {blob.name}: {e}")
                continue  # Skip if timestamp is invalid

            if start_time <= file_timestamp <= now:
                # Download blob content
                csv_content = blob.download_as_string()
                df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
                ecg_data_list.extend(df['ECG Data'].tolist())

        if not ecg_data_list:
            return {"message": "No data available for the specified period."}

        # Process the accumulated data
        ecg_signal = pd.Series(ecg_data_list)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250)

        # Process ECG and detect R-peaks
        processed_data, info = nk.ecg_process(ecg_signal, sampling_rate=250)
        rpeaks = info['ECG_R_Peaks']

        # Calculate heart rate
        heart_rate = nk.ecg_rate(rpeaks, sampling_rate=250)

        # Detect abnormalities
        tachycardia_periods = np.where(heart_rate > 100)[0]
        bradycardia_periods = np.where(heart_rate < 60)[0]
        flatline_periods = np.where(np.diff(rpeaks) > 250 * 1.5)[0]

        # Calculate cardiac score
        total_periods = len(heart_rate)
        abnormal_periods = len(tachycardia_periods) + len(bradycardia_periods) + len(flatline_periods)
        cardiac_score = 1 - (abnormal_periods / total_periods)

        return {"cardiac_score": cardiac_score}
    except Exception as e:
        logging.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating statistics: {e}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
