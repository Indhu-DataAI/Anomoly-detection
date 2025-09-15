from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import io
import json
from typing import Dict, List, Any
from pydantic import BaseModel

app = FastAPI(
    title="Smart System Anomaly Detection API",
    description="API for detecting anomalies in smart system data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
scaler = None
label_encoder = None
label_encoder1 = None
feature_columns = None
feature_importance = None

class PredictionResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str

class ModelInfo(BaseModel):
    model_type: str
    features_count: int
    classes: List[str]
    required_features: List[str]

@app.on_event("startup")
async def load_model_components():
    """Load all trained model components on startup"""
    global model, scaler, label_encoder, label_encoder1, feature_columns, feature_importance
    
    try:
        model = joblib.load('models/xgb_anomaly_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        label_encoder1 = joblib.load('models/label_encoder1.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        feature_importance = pd.read_csv('models/feature_importance.csv')
        
        print("SUCCESS: All model components loaded successfully")
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found: {e}")
        raise HTTPException(status_code=500, detail=f"Model files not found: {e}")

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess uploaded data for prediction"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    df_processed = df.copy()
    
    # Encode device_type if exists
    if 'device_type' in df_processed.columns:
        try:
            df_processed['device_type'] = label_encoder.transform(df_processed['device_type'])
        except ValueError:
            # Handle unknown device types
            unknown_mask = ~df_processed['device_type'].isin(label_encoder.classes_)
            if unknown_mask.any():
                most_frequent_class = df_processed['device_type'].mode().iloc[0] if len(df_processed['device_type'].mode()) > 0 else label_encoder.classes_[0]
                df_processed.loc[unknown_mask, 'device_type'] = most_frequent_class
            df_processed['device_type'] = label_encoder.transform(df_processed['device_type'])
    
    # Keep only training features
    feature_data = df_processed[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(feature_data)
    return X_scaled, feature_data

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {"message": "Smart System Anomaly Detection API", "status": "running"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return ModelInfo(
        model_type="XGBoost Classifier",
        features_count=len(feature_columns),
        classes=label_encoder1.classes_.tolist(),
        required_features=feature_columns.tolist()
    )

@app.get("/feature-importance", tags=["Model"])
async def get_feature_importance():
    """Get feature importance data"""
    if feature_importance is None:
        raise HTTPException(status_code=500, detail="Feature importance not loaded")
    
    return {
        "feature_importance": feature_importance.head(10).to_dict('records')
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_anomalies(file: UploadFile = File(...)):
    """Upload CSV file and get anomaly predictions"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Check for required features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}"
            )
        
        # Preprocess and predict
        X_scaled, _ = preprocess_data(df)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Decode predictions
        decoded_predictions = label_encoder1.inverse_transform(predictions)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Add results to dataframe
        df['prediction'] = decoded_predictions
        df['confidence'] = confidence_scores
        
        # Calculate prediction statistics
        pred_counts = pd.Series(decoded_predictions).value_counts()
        total_records = len(df)
        
        prediction_stats = []
        for class_name, count in pred_counts.items():
            prediction_stats.append({
                'class': class_name,
                'count': int(count),
                'percentage': round((count / total_records) * 100, 1)
            })
        
        # Prepare response data
        response_data = {
            'total_records': total_records,
            'prediction_stats': prediction_stats,
            'predictions': df.to_dict('records'),
            'confidence_distribution': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            }
        }
        
        return PredictionResponse(
            success=True,
            data=response_data,
            message="Predictions completed successfully"
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-sample", tags=["Prediction"])
async def predict_sample_data(data: Dict[str, Any]):
    """Predict anomalies for sample JSON data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert JSON to DataFrame
        if 'records' in data:
            df = pd.DataFrame(data['records'])
        else:
            df = pd.DataFrame([data])
        
        # Check for required features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}"
            )
        
        # Preprocess and predict
        X_scaled, _ = preprocess_data(df)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Decode predictions
        decoded_predictions = label_encoder1.inverse_transform(predictions)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Prepare results
        results = []
        for i in range(len(df)):
            result = df.iloc[i].to_dict()
            result['prediction'] = decoded_predictions[i]
            result['confidence'] = float(confidence_scores[i])
            results.append(result)
        
        return {
            'success': True,
            'predictions': results,
            'message': 'Sample prediction completed successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)