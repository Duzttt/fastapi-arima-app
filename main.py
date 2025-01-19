from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd
import datetime as dt
from model import ARIMAModel
import matplotlib.pyplot as plt
import io
import base64
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount the static files directory to serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize ARIMA model with correct path
model_path = "/app/static/arima_model.pkl"
try:
    model = ARIMAModel(model_path=model_path)
    model.load_model()  # Load pre-trained ARIMA model
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Define input schema for prediction
class PredictRequest(BaseModel):
    month: int
    year: int

    @validator("month")
    def validate_month(cls, v):
        if v < 1 or v > 12:
            raise ValueError("Month must be between 1 and 12.")
        return v

    @validator("year")
    def validate_year(cls, v):
        current_year = dt.datetime.now().year
        if v < current_year:
            raise ValueError("Year must be equal to or greater than the current year.")
        return v

@app.get("/")
def read_root():
    """
    Root endpoint to serve the HTML file.
    """
    return FileResponse("static/index.html")

@app.post("/predict/")
def predict(request: PredictRequest):
    """
    Predict future values for a given month and year.
    """
    try:
        logger.info(f"Received prediction request: {request}")
        
        # Current date
        current_date = dt.datetime.now()

        # Target date from user input
        target_date = dt.datetime(year=request.year, month=request.month, day=1)

        # Calculate the number of steps (months) into the future
        if target_date <= current_date:
            raise HTTPException(
                status_code=400, detail="Target date must be in the future."
            )

        # Calculate months difference
        steps = (target_date.year - current_date.year) * 12 + (target_date.month - current_date.month)

        # Predict future values
        predictions = model.predict(steps=steps)
        logger.info(f"Predictions: {predictions}")

        # Plot the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, steps + 1), predictions, label="Predicted Values")
        plt.title(f"ARIMA Predictions for {request.month}/{request.year}")
        plt.xlabel("Steps (Months)")
        plt.ylabel("Predicted Values")
        plt.legend()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Encode the plot to base64
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {
            "target_date": f"{request.month}/{request.year}",
            "steps_into_future": steps,
            "predictions": predictions,
            "plot": plot_base64,
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
