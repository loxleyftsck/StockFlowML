import logging
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionInput, PredictionOutput, HealthCheck
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
MODEL_PATH = Path("models/logistic_model.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup and clean up on shutdown.
    """
    global model
    logger.info("Loading model...")
    try:
        if MODEL_PATH.exists():
            # Load dictionary containing model, scaler, etc.
            model_artifact = joblib.load(MODEL_PATH)
            model = model_artifact # Store the whole dict
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            logger.info(f"Model type: {model.get('model_type', 'unknown')}")
            logger.info(f"Features: {model.get('feature_names', [])}")
        else:
            logger.error(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API...")

app = FastAPI(
    title="StockFlowML API",
    description="Production-grade API for Stock Trend Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# ... middleware code ...

@app.get("/", tags=["Status"])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to StockFlowML API. Visit /docs for documentation."}

@app.get("/health", response_model=HealthCheck, tags=["Status"])
async def health_check():
    """Check API health and model status."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict(input_data: PredictionInput):
    """
    Make a prediction based on provided features.
    """
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract components from loaded artifact
        sklearn_model = model['model']
        scaler = model['scaler']
        feature_names = model.get('feature_names', [])
        
        # fallback if feature_names not found
        if not feature_names:
            feature_names = ['returns', 'ma_5', 'ma_20', 'volatility_5', 'volatility_20']
        
        # Prepare input data
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Sort/Filter columns to match training order
        try:
            prediction_df = input_df[feature_names]
        except KeyError as e:
            missing = set(feature_names) - set(input_df.columns)
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
            
        # Scale features using the saved scaler
        X_scaled = scaler.transform(prediction_df)
        
        # Predict
        prediction = sklearn_model.predict(X_scaled)[0]
        probability = sklearn_model.predict_proba(X_scaled)[0][1]
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_version": model.get('trained_at', 'unknown'),
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
