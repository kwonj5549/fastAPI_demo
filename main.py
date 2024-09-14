from fastapi import FastAPI, HTTPException, Query, Request
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
    timestamp: str = None  # Optional timestamp in ISO 8601 format

class ECGOutputData(BaseModel):
    results: dict
    cardiac_score: float

# Store accumulated data in a global variable
accumulated_ecg_data = []

# GCP Bucket name and path
BUCKET_NAME = 'fastapibucket'  # Replace with your bucket name
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
def process_ecg_file(data: ECGInputData) -> ECGOutputData:
    try:
        # Parse timestamp and ensure it's timezone-aware
        if data.timestamp:
            try:
                timestamp_parsed = pd.Timestamp(data.timestamp)
                if timestamp_parsed.tzinfo is None:  # If tz-naive, localize to UTC
                    timestamp_parsed = timestamp_parsed.tz_localize('UTC')
                else:
                    timestamp_parsed = timestamp_parsed.tz_convert('UTC')
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 format.")
        else:
            timestamp_parsed = pd.Timestamp.now(tz='UTC')

        timestamp_str = timestamp_parsed.isoformat()
        timestamp_for_filename = timestamp_parsed.strftime('%Y%m%d_%H%M%S')

        ecg_signal = pd.Series(data.ecg_data)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250)

        # Process ECG and detect R-peaks
        processed_data, info = nk.ecg_process(ecg_signal, sampling_rate=250)
        rpeaks = info['ECG_R_Peaks']

        # Calculate heart rate
        heart_rate = nk.ecg_rate(rpeaks, sampling_rate=250, desired_length=len(ecg_signal))

        # Detect flatline periods
        flatline_threshold = 250 * 1.5  # No R-peaks for more than 1.5 seconds
        flatline_indices = np.where(np.diff(rpeaks) > flatline_threshold)[0]

        # Initialize a boolean array for flatline periods
        is_flatline = np.zeros(len(heart_rate), dtype=bool)

        # Mark flatline periods in the boolean array
        for idx in flatline_indices:
            start = rpeaks[idx]
            end = rpeaks[idx + 1] if idx + 1 < len(rpeaks) else len(ecg_signal)
            is_flatline[start:end] = True

        # Detect tachycardia (exclude flatline periods)
        tachycardia_indices = np.where((heart_rate > 100) & (~is_flatline))[0]

        # Detect bradycardia (exclude flatline periods)
        bradycardia_indices = np.where((heart_rate < 60) & (~is_flatline))[0]

        # Convert periods to start and end times
        def get_periods(indices):
            periods = []
            for idx in indices:
                if idx + 1 < len(ecg_signal):
                    periods.append([idx / 250, (idx + 1) / 250])
            return periods

        # Merge overlapping intervals
        merged_tachycardia = merge_intervals(get_periods(tachycardia_indices.tolist()))
        merged_bradycardia = merge_intervals(get_periods(bradycardia_indices.tolist()))
        merged_flatline = []
        for idx in flatline_indices:
            start_time = rpeaks[idx] / 250
            end_time = rpeaks[idx + 1] / 250 if idx + 1 < len(rpeaks) else len(ecg_signal) / 250
            merged_flatline.append([start_time, end_time])
        merged_flatline = merge_intervals(merged_flatline)

        # Calculate cardiac score
        total_periods = len(heart_rate)
        abnormal_periods = len(tachycardia_indices) + len(bradycardia_indices) + len(flatline_indices)
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
        request: Request
) -> ECGOutputData:
    global accumulated_ecg_data
    try:
        body = await request.json()
        # Handle the end-of-stream signal and process data
        if "end_of_stream" in body and body["end_of_stream"]:
            # Parse timestamp and ensure it's timezone-aware
            timestamp = body.get("timestamp")
            if timestamp:
                try:
                    timestamp_parsed = pd.Timestamp(timestamp)
                    if timestamp_parsed.tzinfo is None:  # If tz-naive, localize to UTC
                        timestamp_parsed = timestamp_parsed.tz_localize('UTC')
                    else:
                        timestamp_parsed = timestamp_parsed.tz_convert('UTC')
                except Exception as e:
                    raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO 8601 format.")
            else:
                timestamp_parsed = pd.Timestamp.now(tz='UTC')  # Ensure timestamp is timezone-aware

            timestamp_str = timestamp_parsed.isoformat()
            timestamp_for_filename = timestamp_parsed.strftime('%Y%m%d_%H%M%S')

            # Process accumulated data
            ecg_signal = pd.Series(accumulated_ecg_data)
            ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250)

            # Process ECG and detect R-peaks
            processed_data, info = nk.ecg_process(ecg_signal, sampling_rate=250)
            rpeaks = info['ECG_R_Peaks']

            # Calculate heart rate
            heart_rate = nk.ecg_rate(rpeaks, sampling_rate=250, desired_length=len(ecg_signal))

            # Detect flatline periods
            flatline_threshold = 250 * 1.5  # No R-peaks for more than 1.5 seconds
            flatline_indices = np.where(np.diff(rpeaks) > flatline_threshold)[0]

            # Initialize a boolean array for flatline periods
            is_flatline = np.zeros(len(heart_rate), dtype=bool)

            # Mark flatline periods in the boolean array
            for idx in flatline_indices:
                start = rpeaks[idx]
                end = rpeaks[idx + 1] if idx + 1 < len(rpeaks) else len(ecg_signal)
                is_flatline[start:end] = True

            # Detect tachycardia (exclude flatline periods)
            tachycardia_indices = np.where((heart_rate > 100) & (~is_flatline))[0]

            # Detect bradycardia (exclude flatline periods)
            bradycardia_indices = np.where((heart_rate < 60) & (~is_flatline))[0]

            # Convert periods to start and end times
            def get_periods(indices):
                periods = []
                for idx in indices:
                    if idx + 1 < len(ecg_signal):
                        periods.append([idx / 250, (idx + 1) / 250])
                return periods

            # Merge overlapping intervals
            merged_tachycardia = merge_intervals(get_periods(tachycardia_indices.tolist()))
            merged_bradycardia = merge_intervals(get_periods(bradycardia_indices.tolist()))
            merged_flatline = []
            for idx in flatline_indices:
                start_time = rpeaks[idx] / 250
                end_time = rpeaks[idx + 1] / 250 if idx + 1 < len(rpeaks) else len(ecg_signal) / 250
                merged_flatline.append([start_time, end_time])
            merged_flatline = merge_intervals(merged_flatline)

            # Calculate cardiac score
            total_periods = len(heart_rate)
            abnormal_periods = len(tachycardia_indices) + len(bradycardia_indices) + len(flatline_indices)
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
        period: str = Query(None, description="Period for statistics: 'daily' or 'weekly'"),
        start_date: str = Query(None, description="Start date in 'YYYY-MM-DD' format"),
        end_date: str = Query(None, description="End date in 'YYYY-MM-DD' format")
) -> dict:
    try:
        now = pd.Timestamp.now(tz='UTC')  # Ensure current time is tz-aware

        # Parse start_date and end_date
        if start_date and end_date:
            try:
                start_time = pd.Timestamp(start_date)
                end_time = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid date format. Use 'YYYY-MM-DD'.")
        elif start_date:
            try:
                start_time = pd.Timestamp(start_date)
                end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use 'YYYY-MM-DD'.")
        elif end_date:
            try:
                end_time = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                start_time = end_time - pd.Timedelta(days=1) + pd.Timedelta(seconds=1)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use 'YYYY-MM-DD'.")
        else:
            if period == 'daily':
                start_time = now - pd.Timedelta(days=1)
            elif period == 'weekly':
                start_time = now - pd.Timedelta(days=7)
            else:
                raise HTTPException(status_code=400, detail="Provide 'start_date' and 'end_date' or set 'period' to 'daily' or 'weekly'.")
            end_time = now

        # Ensure start_time and end_time are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC')
        else:
            start_time = start_time.tz_convert('UTC')
        if end_time.tzinfo is None:
            end_time = end_time.tz_localize('UTC')
        else:
            end_time = end_time.tz_convert('UTC')

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
                if file_timestamp.tzinfo is None:
                    file_timestamp = file_timestamp.tz_localize('UTC')
                else:
                    file_timestamp = file_timestamp.tz_convert('UTC')
            except Exception as e:
                logging.warning(f"Invalid timestamp in metadata for blob {blob.name}: {e}")
                continue  # Skip if timestamp is invalid

            if start_time <= file_timestamp <= end_time:
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
        tachycardia_indices = np.where(heart_rate > 100)[0]
        bradycardia_indices = np.where(heart_rate < 60)[0]
        flatline_indices = np.where(np.diff(rpeaks) > 250 * 1.5)[0]

        # Calculate cardiac score
        total_periods = len(heart_rate)
        abnormal_periods = len(tachycardia_indices) + len(bradycardia_indices) + len(flatline_indices)
        cardiac_score = 1 - (abnormal_periods / total_periods)

        # **Modified here to only return the cardiac score**
        return {"cardiac_score": cardiac_score}

    except Exception as e:
        logging.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating statistics: {e}")

@app.get("/")
def read_root():
    return {"Hello": "World"}
