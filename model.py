import pickle
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """Load a pre-trained ARIMA model from a pickle file."""
        if self.model_path:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise ValueError("Model path not provided.")

    def save_model(self, model_path):
        """Save the trained ARIMA model to a pickle file."""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self, data, order=(1, 1, 1)):
        """Train an ARIMA model with the given data and order."""
        self.model = ARIMA(data, order=order).fit()
        return self.model

    def predict(self, steps):
        """Generate forecasts using the trained model."""
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
        forecast = self.model.forecast(steps=steps)
        return forecast.tolist()
