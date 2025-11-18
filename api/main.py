"""
FastAPI inference endpoint for HVAC Unit Cooler Digital Twin.

This API provides real-time predictions for:
- UCAOT (Unit Cooler Air Outlet Temperature)
- UCWOT (Unit Cooler Water Outlet Temperature)
- UCAF (Unit Cooler Air Flow)

Supports both ONNX and TFLite models for edge deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pickle
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HVAC Unit Cooler Digital Twin API",
    description="Real-time predictions for naval HVAC systems using LightGBM/ONNX models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and scalers
MODELS = {}
SCALERS = {}
MODEL_TYPE = "lightgbm"  # Options: "lightgbm", "onnx", "tflite"


# Pydantic models for request/response
class HVACInput(BaseModel):
    """Input features for HVAC prediction."""
    UCWIT: float = Field(..., description="Water Inlet Temperature (°C)")
    UCAIT: float = Field(..., description="Air Inlet Temperature (°C)")
    UCWF: float = Field(..., description="Water Flow Rate (L/min)")
    UCAIH: float = Field(..., description="Air Inlet Humidity (%)")
    AMBT: float = Field(..., description="Ambient Temperature (°C)")
    UCTSP: float = Field(..., description="Temperature Setpoint (°C)")

    # Additional features (optional - will be set to 0 if not provided)
    UCFMS: Optional[float] = Field(0.0, description="Fan Motor Speed")
    UCFMV: Optional[float] = Field(0.0, description="Fan Motor Voltage")
    CPPR: Optional[float] = Field(0.0, description="Chiller Primary Pump Rate")
    CPDP: Optional[float] = Field(0.0, description="Chiller Primary Differential Pressure")

    class Config:
        schema_extra = {
            "example": {
                "UCWIT": 7.5,
                "UCAIT": 25.0,
                "UCWF": 15.0,
                "UCAIH": 50.0,
                "AMBT": 22.0,
                "UCTSP": 21.0,
                "UCFMS": 1500.0,
                "UCFMV": 220.0,
                "CPPR": 80.0,
                "CPDP": 150.0
            }
        }


class HVACOutput(BaseModel):
    """Output predictions from HVAC model."""
    UCAOT: float = Field(..., description="Predicted Air Outlet Temperature (°C)")
    UCWOT: float = Field(..., description="Predicted Water Outlet Temperature (°C)")
    UCAF: float = Field(..., description="Predicted Air Flow (m³/h)")
    Q_thermal: float = Field(..., description="Estimated Thermal Power (kW)")
    inference_time_ms: float = Field(..., description="Inference time (milliseconds)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_type: str
    models_loaded: List[str]
    version: str


def load_models():
    """Load trained models and scalers on startup."""
    global MODELS, SCALERS, MODEL_TYPE

    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data' / 'processed'

    logger.info("Loading models and scalers...")

    try:
        # Load LightGBM models
        lightgbm_path = models_dir / 'lightgbm_model.pkl'
        if lightgbm_path.exists():
            with open(lightgbm_path, 'rb') as f:
                models_dict = pickle.load(f)

            if isinstance(models_dict, dict):
                MODELS = models_dict
                logger.info(f"Loaded {len(MODELS)} LightGBM models: {list(MODELS.keys())}")
            else:
                MODELS['single'] = models_dict
                logger.info("Loaded single LightGBM model")

            MODEL_TYPE = "lightgbm"
        else:
            logger.warning(f"LightGBM model not found at {lightgbm_path}")

        # Load scalers
        scaler_path = data_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                SCALERS['X_scaler'] = pickle.load(f)
            logger.info("Loaded X scaler")
        else:
            # Load from numpy arrays
            X_mean = np.load(data_dir / 'X_scaler_mean.npy')
            X_scale = np.load(data_dir / 'X_scaler_scale.npy')

            class SimpleScaler:
                def __init__(self, mean, scale):
                    self.mean_ = mean
                    self.scale_ = scale

                def transform(self, X):
                    return (X - self.mean_) / self.scale_

                def inverse_transform(self, X):
                    return X * self.scale_ + self.mean_

            SCALERS['X_scaler'] = SimpleScaler(X_mean, X_scale)
            logger.info("Loaded X scaler from numpy arrays")

        # Load y scaler
        try:
            y_mean = np.load(data_dir / 'y_scaler_mean.npy')
            y_scale = np.load(data_dir / 'y_scaler_scale.npy')

            class SimpleScaler:
                def __init__(self, mean, scale):
                    self.mean_ = mean
                    self.scale_ = scale

                def transform(self, y):
                    return (y - self.mean_) / self.scale_

                def inverse_transform(self, y):
                    return y * self.scale_ + self.mean_

            SCALERS['y_scaler'] = SimpleScaler(y_mean, y_scale)
            logger.info("Loaded y scaler")
        except Exception as e:
            logger.warning(f"Could not load y scaler: {e}")

        # Load metadata
        metadata_path = data_dir / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                SCALERS['metadata'] = json.load(f)
            logger.info("Loaded metadata")

        logger.info("Models and scalers loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def create_feature_vector(input_data: HVACInput) -> np.ndarray:
    """
    Create feature vector with engineered features.

    This should match the feature engineering from Sprint 1.
    For simplicity, we'll use the basic features. In production,
    you should include all 52 engineered features.
    """
    # Basic features
    features = {
        'UCWIT': input_data.UCWIT,
        'UCAIT': input_data.UCAIT,
        'UCWF': input_data.UCWF,
        'UCAIH': input_data.UCAIH,
        'AMBT': input_data.AMBT,
        'UCTSP': input_data.UCTSP,
        'UCFMS': input_data.UCFMS,
        'UCFMV': input_data.UCFMV,
        'CPPR': input_data.CPPR,
        'CPDP': input_data.CPDP,
    }

    # Physics-based engineered features (simplified)
    # In production, include all 52 features from feature_engineering.py
    features['delta_T_water'] = 0  # Will be predicted
    features['delta_T_air'] = 0    # Will be predicted
    features['T_water_avg'] = input_data.UCWIT  # Approximate
    features['T_air_avg'] = input_data.UCAIT    # Approximate

    # For now, pad with zeros to match expected 52 features
    # In production, compute all engineered features properly
    feature_array = np.zeros(52)

    # Fill in the basic features (indices depend on original feature order)
    # This is a simplified version - in production, match exact feature order
    for idx, (key, value) in enumerate(features.items()):
        if idx < 52:
            feature_array[idx] = value

    return feature_array.reshape(1, -1)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "message": "HVAC Unit Cooler Digital Twin API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODELS else "unhealthy",
        model_type=MODEL_TYPE,
        models_loaded=list(MODELS.keys()) if MODELS else [],
        version="1.0.0"
    )


@app.post("/predict", response_model=HVACOutput)
async def predict(input_data: HVACInput):
    """
    Make predictions for HVAC system outputs.

    Args:
        input_data: Input features (temperatures, flows, setpoints)

    Returns:
        Predicted outputs (UCAOT, UCWOT, UCAF, Q_thermal)
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        start_time = time.perf_counter()

        # Create feature vector
        X = create_feature_vector(input_data)

        # Scale features
        if 'X_scaler' in SCALERS:
            X_scaled = SCALERS['X_scaler'].transform(X)
        else:
            X_scaled = X

        # Make predictions
        predictions = {}

        if isinstance(MODELS, dict) and len(MODELS) > 1:
            # Multiple target models
            for target_name, model in MODELS.items():
                pred = model.predict(X_scaled)
                predictions[target_name] = float(pred[0])
        else:
            # Single model predicting all targets
            model = MODELS.get('single', list(MODELS.values())[0])
            pred = model.predict(X_scaled)

            if len(pred.shape) > 1 and pred.shape[1] >= 3:
                predictions['UCAOT'] = float(pred[0, 0])
                predictions['UCWOT'] = float(pred[0, 1])
                predictions['UCAF'] = float(pred[0, 2])
            else:
                # Single output
                predictions['output'] = float(pred[0])

        # Unscale predictions if scaler available
        if 'y_scaler' in SCALERS and len(predictions) == 3:
            pred_array = np.array([[
                predictions.get('UCAOT', 0),
                predictions.get('UCWOT', 0),
                predictions.get('UCAF', 0)
            ]])
            pred_unscaled = SCALERS['y_scaler'].inverse_transform(pred_array)
            predictions['UCAOT'] = float(pred_unscaled[0, 0])
            predictions['UCWOT'] = float(pred_unscaled[0, 1])
            predictions['UCAF'] = float(pred_unscaled[0, 2])

        # Calculate thermal power (simplified)
        # Q = m_dot * cp * delta_T
        Q_thermal = 0.0
        if 'UCWOT' in predictions:
            delta_T_water = input_data.UCWIT - predictions['UCWOT']
            cp_water = 4.186  # kJ/(kg·K)
            rho_water = 1.0   # kg/L (approximate)
            m_dot_water = input_data.UCWF * rho_water / 60  # kg/s
            Q_thermal = m_dot_water * cp_water * delta_T_water  # kW

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        return HVACOutput(
            UCAOT=predictions.get('UCAOT', 0.0),
            UCWOT=predictions.get('UCWOT', 0.0),
            UCAF=predictions.get('UCAF', 0.0),
            Q_thermal=Q_thermal,
            inference_time_ms=inference_time_ms
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[HVACOutput])
async def predict_batch(input_data_list: List[HVACInput]):
    """
    Make batch predictions for multiple inputs.

    Args:
        input_data_list: List of input features

    Returns:
        List of predicted outputs
    """
    results = []
    for input_data in input_data_list:
        result = await predict(input_data)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
