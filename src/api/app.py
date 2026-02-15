import logging
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from .schemas import PredictionInput, PredictionOutput, HealthCheck, TickerPredictionInput
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
prediction_counter = Counter(
    'stockflowml_prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)

prediction_latency = Histogram(
    'stockflowml_prediction_latency_seconds',
    'Prediction request latency in seconds',
    ['endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

model_predictions = Counter(
    'stockflowml_model_predictions_total',
    'Total predictions by class',
    ['prediction_class']
)

feature_store_latency = Histogram(
    'stockflowml_feature_store_latency_seconds',
    'Feature store query latency in seconds',
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
)

model_info = Gauge(
    'stockflowml_model_info',
    'Model metadata',
    ['model_type', 'version']
)

# Global Feast store
feast_store = None

def initialize_feast():
    """Initialize Feast feature store (optional)."""
    global feast_store
    try:
        from feast import FeatureStore
        from pathlib import Path
        
        repo_path = Path(__file__).parent.parent.parent / "feature_store" / "feature_repo"
        if repo_path.exists():
            feast_store = FeatureStore(repo_path=str(repo_path))
            logger.info("✓ Feast feature store initialized")
            return True
    except Exception as e:
        logger.warning(f"Feast not available: {e}")
    return False

# Try to initialize Feast on startup
feast_enabled = initialize_feast()

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
        "model_loaded": model is not None,
        "feast_enabled": feast_store is not None
    }

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict(input_data: PredictionInput):
    """
    Make a prediction based on provided features.
    """
    start_time = time.time()
    
    if model is None:
        prediction_counter.labels(endpoint='/predict', status='error').inc()
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
            prediction_counter.labels(endpoint='/predict', status='error').inc()
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
            
        # Scale features using the saved scaler
        X_scaled = scaler.transform(prediction_df)
        
        # Predict
        prediction = sklearn_model.predict(X_scaled)[0]
        probability = sklearn_model.predict_proba(X_scaled)[0][1]
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update Prometheus metrics
        prediction_counter.labels(endpoint='/predict', status='success').inc()
        prediction_latency.labels(endpoint='/predict').observe(time.time() - start_time)
        
        # Track prediction class distribution
        prediction_class = 'up' if prediction == 1 else 'down'
        model_predictions.labels(prediction_class=prediction_class).inc()
        
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
        prediction_counter.labels(endpoint='/predict', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/ticker", response_model=PredictionOutput, tags=["Inference"])
async def predict_from_ticker(input_data: TickerPredictionInput):
    """
    Make a prediction by fetching features from Feast feature store.
    Requires Feast to be initialized and features to be materialized.
    """
    from datetime import datetime
    
    start_time = time.time()
    
    if model is None:
        prediction_counter.labels(endpoint='/predict/ticker', status='error').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if feast_store is None:
        prediction_counter.labels(endpoint='/predict/ticker', status='error').inc()
        raise HTTPException(
            status_code=503, 
            detail="Feast feature store not available. Use /predict endpoint with manual features."
        )
    
    try:
        # Prepare entity dataframe for Feast
        timestamp = input_data.timestamp or datetime.now()
        entity_df = pd.DataFrame({
            "ticker": [input_data.ticker],
            "event_timestamp": [timestamp]
        })
        
        # Fetch features from Feast
        feast_start = time.time()
        feature_vector = feast_store.get_online_features(
            features=[
                "stock_technical_features:returns",
                "stock_technical_features:ma_5",
                "stock_technical_features:ma_10",
                "stock_technical_features:ma_20",
                "stock_technical_features:volatility_5",
                "stock_technical_features:volatility_10",
                "stock_technical_features:volatility_20",
            ],
            entity_rows=entity_df.to_dict('records')
        ).to_dict()
        feast_latency_time = time.time() - feast_start
        feature_store_latency.observe(feast_latency_time)
        
        # Convert to DataFrame
        feature_names = model.get('feature_names', [])
        if not feature_names:
            feature_names = ['returns', 'ma_5', 'ma_10', 'ma_20', 'volatility_5', 'volatility_10', 'volatility_20']
        
        # Extract feature values
        feature_values = {}
        for fname in feature_names:
            key = f"stock_technical_features__{fname}"
            if key in feature_vector:
                feature_values[fname] = feature_vector[key][0]
        
        if not feature_values:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for ticker {input_data.ticker} at {timestamp}"
            )
        
        prediction_df = pd.DataFrame([feature_values])
        
        # Scale and predict
        sklearn_model = model['model']
        scaler = model['scaler']
        
        X_scaled = scaler.transform(prediction_df)
        prediction = sklearn_model.predict(X_scaled)[0]
        probability = sklearn_model.predict_proba(X_scaled)[0][1]
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update Prometheus metrics
        prediction_counter.labels(endpoint='/predict/ticker', status='success').inc()
        prediction_latency.labels(endpoint='/predict/ticker').observe(time.time() - start_time)
        
        # Track prediction class distribution
        prediction_class = 'up' if prediction == 1 else 'down'
        model_predictions.labels(prediction_class=prediction_class).inc()
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_version": model.get('trained_at', 'unknown'),
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ticker prediction error: {e}")
        prediction_counter.labels(endpoint='/predict/ticker', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
