# backend/main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    # Your logic to save file and trigger preprocessing
    return {"dataset_id": "test_id_123"}

@app.post("/predict")
async def predict(data: dict):
    # Your Deep Sets Model Inference
    return {
        "prediction": "Relapse",
        "confidence": 0.85,
        "probabilities": {"C": 0.1, "L": 0.85, "I": 0.05}
    }

# Add your MOI and Allele Frequency endpoints here...

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)