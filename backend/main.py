from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import plotly.io as pio  # For converting Plotly figures to JSON
from PvDeepSetNet.src.DeepSets import run_prediction # Your packaged model
import uvicorn

app = FastAPI()

# Temporary storage for demonstration
storage = {}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()
    dataset_id = "ds_001" # In a real app, use: str(uuid.uuid4())
    storage[dataset_id] = contents 
    return {"dataset_id": dataset_id}

@app.post("/predict")
async def predict(data: dict):
    dataset_id = data.get("dataset_id")
    
    # 1. Retrieve data from storage
    file_bytes = storage.get(dataset_id)
    if not file_bytes:
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(io.BytesIO(file_bytes))
    
    # 2. Run the real PvDeepSetNet model
    # Note: run_prediction now returns 3 items: table, text, and a plotly figure
    model_output = run_prediction(df)
    
    res_df = model_output['results_table']
    # The figure object generated inside your model processing file
    fig1 = model_output.get('donut_plot') 
    fig2 = model_output.get('distribution_plot') 

    # 3. Serialize the Plotly Figure to JSON
    # This allows the plot to be sent over the internet as a string
    plot_json1 = pio.to_json(fig1) if fig1 else None
    plot_json2 = pio.to_json(fig2) if fig2 else None

    
    return {
        "results_table": res_df.to_dict('records'),
        "plot_json1": plot_json1,  # The serialized figure
        "plot_json2": plot_json2  # The serialized figure
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)