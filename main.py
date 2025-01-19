from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
from model import ARIMAModel
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# Initialize the ARIMA model class
model = ARIMAModel(model_path="arima_model.pkl")

try:
    model.load_model()  # Load the pre-trained model
except Exception as e:
    print(f"Error loading model: {e}")

# Define input schemas
class TrainRequest(BaseModel):
    data: list
    order: tuple

class PredictRequest(BaseModel):
    steps: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the ARIMA Model API!"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and process its data.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file.file, parse_dates=["Date"], index_col="Date")
        # Extract the values for the ARIMA model
        data = df["Gred A"].values.tolist()
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model(request: TrainRequest):
    """
    Train the ARIMA model with provided data and order.
    """
    try:
        trained_model = model.train(data=np.array(request.data), order=request.order)
        model.save_model("arima_model.pkl")  # Save the trained model
        return {"message": "Model trained successfully", "summary": trained_model.summary().as_text()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict future values using the trained ARIMA model.
    """
    try:
        predictions = model.predict(steps=request.steps)
        
        # Plot the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(predictions, label="Predictions")
        plt.legend()
        plt.title("ARIMA Model Predictions")
        plt.xlabel("Time")
        plt.ylabel("Values")
        
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the plot to base64 string
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {"predictions": predictions, "plot": plot_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
