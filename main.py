import requests
import pandas as pd
import json
import time

# URL of the FastAPI streaming endpoint
url = "http://34.122.213.139:8000/process_ecg_stream"  # Replace with your actual server IP or localhost

# Function to read and chunk data from a CSV file
def read_and_chunk_csv(file_path, chunk_size):
    df = pd.read_csv(file_path)
    ecg_column = 'ECG'  # Replace with your actual column name if different

    for start in range(0, len(df), chunk_size):
        chunk = df[ecg_column][start:start + chunk_size].tolist()
        yield {"ecg_data": chunk}

# Function to stream data to the endpoint
def stream_data(url, data_chunks):
    for chunk in data_chunks:
        response = requests.post(url, json=chunk, headers={"Content-Type": "application/json"})

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response Data:", response.json())
        else:
            print("Error:", response.text)

        time.sleep(0.5)

    # Signal end of stream
    end_signal = {"end_of_stream": True}
    response = requests.post(url, json=end_signal, headers={"Content-Type": "application/json"})

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response Data:", response.json())
    else:
        print("Error:", response.text)

file_path = 'bio_data.csv'  # Replace with the path to your CSV file
chunk_size = 1000  # Define how many data points per chunk

data_chunks = read_and_chunk_csv(file_path, chunk_size)

stream_data(url, data_chunks)
