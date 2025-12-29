"""
🧠 Cogaly API - Alzheimer's Risk Detection Endpoint
====================================================
FastAPI backend for serving Cogaly predictions with SHAP explainability.

Run locally:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict - Get Alzheimer's risk prediction
    GET /health - Health check
    GET /features - Get required features list
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="Cogaly API",
    description="AI-powered early Alzheimer's risk detection with SHAP explainability",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Load Model Artifacts
# ============================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

def load_artifacts():
    """Load all model artifacts"""
    try:
        with open(os.path.join(MODEL_DIR, 'cogaly_xgb_v1.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'cogaly_scaler_v1.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'cogaly_feature_columns_v1.pkl'), 'rb') as f:
            features = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'cogaly_shap_explainer_v1.pkl'), 'rb') as f:
            explainer = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'cogaly_metrics_v1.pkl'), 'rb') as f:
            metrics = pickle.load(f)
        
        return model, scaler, features, explainer, metrics
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None, None, None

# Load models at startup
model, scaler, features, explainer, metrics = load_artifacts()

# ============================================================================
# Request/Response Models
# ============================================================================

class PatientData(BaseModel):
    """Patient input data for prediction"""
    Age: float
    Gender: int  # 0 = Female, 1 = Male
    Ethnicity: int  # 0-3
    EducationLevel: int  # 0-3
    BMI: float
    Smoking: int  # 0 = No, 1 = Yes
    AlcoholConsumption: float  # 0-20 units/week
    PhysicalActivity: float  # 0-10 hours/week
    DietQuality: float  # 0-10 score
    SleepQuality: float  # 4-10 score
    FamilyHistoryAlzheimers: int  # 0 = No, 1 = Yes
    CardiovascularDisease: int  # 0 = No, 1 = Yes
    Diabetes: int  # 0 = No, 1 = Yes
    Depression: int  # 0 = No, 1 = Yes
    HeadInjury: int  # 0 = No, 1 = Yes
    Hypertension: int  # 0 = No, 1 = Yes
    SystolicBP: float  # 90-180 mmHg
    DiastolicBP: float  # 60-120 mmHg
    CholesterolTotal: float  # 150-300 mg/dL
    CholesterolLDL: float  # 50-200 mg/dL
    CholesterolHDL: float  # 20-100 mg/dL
    CholesterolTriglycerides: float  # 50-400 mg/dL
    MMSE: float  # 0-30 (Mini-Mental State Examination)
    FunctionalAssessment: float  # 0-10 score
    MemoryComplaints: int  # 0 = No, 1 = Yes
    BehavioralProblems: int  # 0 = No, 1 = Yes
    ADL: float  # 0-10 (Activities of Daily Living)
    Confusion: int  # 0 = No, 1 = Yes
    Disorientation: int  # 0 = No, 1 = Yes
    PersonalityChanges: int  # 0 = No, 1 = Yes
    DifficultyCompletingTasks: int  # 0 = No, 1 = Yes
    Forgetfulness: int  # 0 = No, 1 = Yes

class PredictionResponse(BaseModel):
    """Prediction response with explainability"""
    diagnosis: str
    risk_score: float
    confidence: float
    risk_level: str
    top_features: List[Dict]
    recommendations: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_accuracy: Optional[float]

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "error",
        model_loaded=model is not None,
        model_accuracy=metrics.get('accuracy') if metrics else None
    )

@app.get("/features")
async def get_features():
    """Get required features and their descriptions"""
    feature_info = {
        "demographics": {
            "Age": {"type": "number", "min": 18, "max": 120, "description": "Patient age in years"},
            "Gender": {"type": "select", "options": [{"value": 0, "label": "Female"}, {"value": 1, "label": "Male"}]},
            "Ethnicity": {"type": "select", "options": [
                {"value": 0, "label": "Caucasian"}, 
                {"value": 1, "label": "African American"}, 
                {"value": 2, "label": "Asian"}, 
                {"value": 3, "label": "Other"}
            ]},
            "EducationLevel": {"type": "select", "options": [
                {"value": 0, "label": "None"}, 
                {"value": 1, "label": "High School"}, 
                {"value": 2, "label": "Bachelor's"}, 
                {"value": 3, "label": "Higher"}
            ]}
        },
        "lifestyle": {
            "BMI": {"type": "number", "min": 15, "max": 50, "description": "Body Mass Index"},
            "Smoking": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "AlcoholConsumption": {"type": "number", "min": 0, "max": 20, "description": "Weekly alcohol units"},
            "PhysicalActivity": {"type": "number", "min": 0, "max": 10, "description": "Weekly exercise hours"},
            "DietQuality": {"type": "number", "min": 0, "max": 10, "description": "Diet quality score"},
            "SleepQuality": {"type": "number", "min": 4, "max": 10, "description": "Sleep quality score"}
        },
        "medical_history": {
            "FamilyHistoryAlzheimers": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "CardiovascularDisease": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "Diabetes": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "Depression": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "HeadInjury": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "Hypertension": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]}
        },
        "clinical": {
            "SystolicBP": {"type": "number", "min": 90, "max": 180, "description": "Systolic blood pressure (mmHg)"},
            "DiastolicBP": {"type": "number", "min": 60, "max": 120, "description": "Diastolic blood pressure (mmHg)"},
            "CholesterolTotal": {"type": "number", "min": 150, "max": 300, "description": "Total cholesterol (mg/dL)"},
            "CholesterolLDL": {"type": "number", "min": 50, "max": 200, "description": "LDL cholesterol (mg/dL)"},
            "CholesterolHDL": {"type": "number", "min": 20, "max": 100, "description": "HDL cholesterol (mg/dL)"},
            "CholesterolTriglycerides": {"type": "number", "min": 50, "max": 400, "description": "Triglycerides (mg/dL)"}
        },
        "cognitive": {
            "MMSE": {"type": "number", "min": 0, "max": 30, "description": "Mini-Mental State Examination score"},
            "FunctionalAssessment": {"type": "number", "min": 0, "max": 10, "description": "Functional assessment score"},
            "ADL": {"type": "number", "min": 0, "max": 10, "description": "Activities of Daily Living score"}
        },
        "behavioral": {
            "MemoryComplaints": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "BehavioralProblems": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "Confusion": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "Disorientation": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "PersonalityChanges": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "DifficultyCompletingTasks": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
            "Forgetfulness": {"type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]}
        }
    }
    return feature_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """Get Alzheimer's risk prediction with explainability"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        patient_dict = patient.dict()
        patient_df = pd.DataFrame([patient_dict])
        
        # Ensure correct feature order
        for col in features:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[features]
        
        # Scale features
        patient_scaled = pd.DataFrame(
            scaler.transform(patient_df),
            columns=features
        )
        
        # Make prediction
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
        risk_score = float(probability[1])
        confidence = float(max(probability) * 100)
        
        # Get SHAP values for explainability
        shap_values = explainer.shap_values(patient_scaled)
        
        # Get top contributing features
        feature_contrib = pd.DataFrame({
            'Feature': features,
            'SHAP_Value': shap_values[0],
            'Value': patient_df.values[0]
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        top_features = [
            {
                'feature': row['Feature'],
                'value': float(row['Value']),
                'shap_value': float(row['SHAP_Value']),
                'impact': 'Increases Risk' if row['SHAP_Value'] > 0 else 'Decreases Risk'
            }
            for _, row in feature_contrib.head(5).iterrows()
        ]
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        # Generate recommendations based on top features
        recommendations = generate_recommendations(top_features, patient_dict)
        
        return PredictionResponse(
            diagnosis="Alzheimer's Risk Detected" if prediction == 1 else "No Significant Risk",
            risk_score=risk_score,
            confidence=confidence,
            risk_level=risk_level,
            top_features=top_features,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def generate_recommendations(top_features: List[Dict], patient_data: Dict) -> List[str]:
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    # Check specific risk factors
    if patient_data.get('Smoking') == 1:
        recommendations.append("Consider smoking cessation programs to reduce vascular risk factors.")
    
    if patient_data.get('PhysicalActivity', 0) < 3:
        recommendations.append("Increase physical activity to at least 150 minutes per week.")
    
    if patient_data.get('SleepQuality', 10) < 6:
        recommendations.append("Improve sleep hygiene for better cognitive health.")
    
    if patient_data.get('MMSE', 30) < 24:
        recommendations.append("Schedule a comprehensive cognitive assessment with a neurologist.")
    
    if patient_data.get('DietQuality', 10) < 5:
        recommendations.append("Consider adopting a Mediterranean or MIND diet.")
    
    # Add general recommendations
    recommendations.append("Maintain regular cognitive and physical activities.")
    recommendations.append("Schedule regular check-ups with your healthcare provider.")
    
    return recommendations[:5]  # Return top 5 recommendations

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

