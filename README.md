# FastAPI ARIMA Prediction Service

This project is a web service built with FastAPI that serves an ARIMA model for time series forecasting. The service allows users to predict future values for a given month and year and visualize the predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Input Validation](#input-validation)
- [Prediction](#prediction)
- [Logging](#logging)

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure the ARIMA model is available at the specified path:**
    Place the pre-trained ARIMA model at `/app/static/arima_model.pkl`.

## Usage

1. **Run the FastAPI application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2. **Access the API documentation:**
    Open your web browser and navigate to `http://0.0.0.0:8000/docs` to view the interactive API documentation.

## API Endpoints

### `GET /`
- **Description:** Serves the HTML file located in the `static` directory.
- **Response:** Returns the `index.html` file.

### `POST /predict/`
- **Description:** Predicts future values for a given month and year.
- **Request Body:**
    ```json
    {
        "month": 1,
        "year": 2025
    }
    ```
- **Response:**
    ```json
    {
        "target_date": "1/2025",
        "steps_into_future": 12,
        "predictions": [predicted_values],
        "plot": "base64_encoded_plot"
    }
    ```

## Input Validation

- **Month Validation:** The month must be between 1 and 12.
- **Year Validation:** The year must be equal to or greater than the current year.

## Prediction

- **Model Loading:** The ARIMA model is loaded from the specified path.
- **Steps Calculation:** The number of steps (months) into the future is calculated based on the current date and the target date provided by the user.
- **Predictions:** Future values are predicted using the ARIMA model.
- **Plotting:** The predictions are plotted and encoded as a base64 string.

## Logging

- Logging is set up to provide information about the model loading process, prediction requests, and any errors encountered.
