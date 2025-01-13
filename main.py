from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model import ARIMAModel

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
        df = pd.read_csv(file.file)
        # Process the DataFrame as needed
        # For example, you can convert it to a list of dictionaries
        data = df.to_dict(orient="records")
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
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
