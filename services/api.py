import requests
import os

BASE_URL = "http://127.0.0.1:8000"  # Update this to your FastAPI port

class PvRecNetAPI:
    @staticmethod
    def upload_dataset(file_bytes, filename):
        files = {'file': (filename, file_bytes, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload-data", files=files)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def run_prediction(dataset_id):
        response = requests.post(f"{BASE_URL}/predict", json={"dataset_id": dataset_id})
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_moi(dataset_id):
        response = requests.post(f"{BASE_URL}/compute-moi", json={"dataset_id": dataset_id})
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_allele_freq(dataset_id):
        response = requests.post(f"{BASE_URL}/allele-frequency", json={"dataset_id": dataset_id})
        response.raise_for_status()
        return response.json()