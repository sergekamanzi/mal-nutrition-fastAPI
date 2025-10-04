from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI(
    title="Malnutrition Risk Prediction API",
    description="API for predicting child stunting risk based on various factors",
    version="1.0.0"
)

# Add CORS middleware to allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
model = None
model_loaded = False

# Create label encoders for categorical variables
label_encoders = {
    'rural_urban': LabelEncoder(),
    'region': LabelEncoder(),
    'mother_education': LabelEncoder()
}

# Define expected categories (you can modify these based on your actual data)
expected_categories = {
    'rural_urban': ['Rural', 'Urban'],
    'region': ['Eastern', 'Western', 'Northern', 'Southern', 'Central'],
    'mother_education': ['None', 'Primary', 'Secondary', 'Higher']
}

def initialize_label_encoders():
    """Initialize label encoders with expected categories"""
    for col, categories in expected_categories.items():
        label_encoders[col].fit(categories)
    print("✅ Label encoders initialized successfully!")

def create_dummy_model():
    """Create a dummy model for testing"""
    global model, model_loaded
    try:
        print("Creating dummy model for testing...")
        
        # Create a simple dummy Random Forest model
        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=10
        )
        
        # Create realistic dummy training data with EXACTLY 15 features
        n_samples = 200
        
        # Create feature matrix with exactly 15 features
        np.random.seed(42)
        X_dummy = np.random.rand(n_samples, 15)
        
        # Make features more realistic
        # Feature 0: Age (0-59 months)
        X_dummy[:, 0] = np.random.randint(0, 60, n_samples)
        # Feature 1: Household income (realistic range)
        X_dummy[:, 1] = np.random.uniform(10000, 300000, n_samples)
        # Feature 2: Family size
        X_dummy[:, 2] = np.random.randint(1, 10, n_samples)
        # Feature 3: Food insecurity (0-1)
        X_dummy[:, 3] = np.random.uniform(0, 1, n_samples)
        # Features 4-9: Binary features (breastfeeding, vaccination, etc.)
        X_dummy[:, 4:10] = np.random.randint(0, 2, (n_samples, 6))
        # Feature 10: Stunting risk score (0-100)
        X_dummy[:, 10] = np.random.uniform(0, 100, n_samples)
        # Features 11-14: Other numerical features (WASH score, etc.)
        X_dummy[:, 11:15] = np.random.uniform(0, 10, (n_samples, 4))
        
        # Create target variable with realistic distribution
        y_dummy = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
        
        # Train the model
        model.fit(X_dummy, y_dummy)
        
        print("✅ Dummy model created and trained successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Model expects {model.n_features_in_} features")
        print(f"Model classes: {model.classes_}")
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error creating dummy model: {e}")
        return False

def load_model():
    """Try to load the actual model file"""
    global model, model_loaded
    model_file_path = 'best_malnutrition_model_random_forest.pkl'
    
    try:
        print(f"Looking for model file: {model_file_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {[f for f in os.listdir('.') if os.path.isfile(f)]}")
        
        if os.path.exists(model_file_path):
            print(f"✅ Found model file: {model_file_path}")
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)
            print("✅ Model loaded successfully!")
            print(f"Model type: {type(model).__name__}")
            if hasattr(model, 'n_features_in_'):
                print(f"Model expects {model.n_features_in_} features")
            if hasattr(model, 'classes_'):
                print(f"Model classes: {model.classes_}")
            model_loaded = True
            return True
        else:
            print(f"❌ Model file not found: {model_file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Initialize label encoders and model
initialize_label_encoders()
if not load_model():
    print("Creating dummy model for testing purposes...")
    create_dummy_model()

# Define input data model
class PredictionInput(BaseModel):
    age_months: float
    household_income: float
    family_size: int
    food_insecurity: float
    breastfeeding: int
    vaccination_complete: int
    diarrhea_last_week: int
    clean_water_access: int
    improved_sanitation: int
    stunting_risk_score: float
    rural_urban: str
    region: str
    mother_education: str

    class Config:
        schema_extra = {
            "example": {
                "age_months": 45.0,
                "household_income": 200000.0,
                "family_size": 6,
                "food_insecurity": 0.7,
                "breastfeeding": 0,
                "vaccination_complete": 1,
                "diarrhea_last_week": 0,
                "clean_water_access": 0,
                "improved_sanitation": 1,
                "stunting_risk_score": 67.0,
                "rural_urban": "Urban",
                "region": "Eastern",
                "mother_education": "Higher"
            }
        }

class PredictionResponse(BaseModel):
    risk_category: str
    risk_probability: float
    confidence: float
    input_validation_notes: list
    risk_factors: list
    manual_risk_score: float
    model_type: str

def validate_user_input(user_data: Dict[str, Any]) -> list:
    """
    Validate user input and provide warnings for unusual values
    """
    warnings = []

    # Age validation (typical stunting monitoring: 0-59 months)
    if user_data['age_months'] > 59:
        warnings.append(f"Age {user_data['age_months']} months is outside typical stunting monitoring range (0-59 months)")

    # Food insecurity validation
    if user_data['food_insecurity'] > 0.7:
        warnings.append("High food insecurity level detected")
    elif user_data['food_insecurity'] < 0.1:
        warnings.append("Very low food insecurity level")

    # WASH score validation
    wash_score = user_data['clean_water_access'] + user_data['improved_sanitation']
    if wash_score == 0:
        warnings.append("Critical: No clean water or sanitation access")
    elif wash_score == 2:
        warnings.append("Excellent WASH conditions")

    # Stunting risk score validation
    if user_data['stunting_risk_score'] > 50:
        warnings.append(f"High clinical risk score ({user_data['stunting_risk_score']}%)")
    elif user_data['stunting_risk_score'] < 20:
        warnings.append(f"Low clinical risk score ({user_data['stunting_risk_score']}%)")

    # Validate categorical values
    if user_data['rural_urban'] not in expected_categories['rural_urban']:
        warnings.append(f"Unusual rural_urban value: {user_data['rural_urban']}")
    
    if user_data['region'] not in expected_categories['region']:
        warnings.append(f"Unusual region value: {user_data['region']}")
    
    if user_data['mother_education'] not in expected_categories['mother_education']:
        warnings.append(f"Unusual mother_education value: {user_data['mother_education']}")

    return warnings

def calculate_risk_factors(user_data: Dict[str, Any]) -> tuple:
    """
    Calculate and weight risk factors more comprehensively
    """
    risk_factors = []
    risk_score = 0

    # Economic factors (weight: 25%)
    income_per_capita = user_data['household_income'] / user_data['family_size']
    if income_per_capita < 20000:  # Example threshold
        risk_factors.append(f"Poverty: income per capita {income_per_capita:,.0f} RWF")
        risk_score += 0.25

    # Food security (weight: 25%)
    if user_data['food_insecurity'] > 0.5:
        risk_factors.append(f"Food insecurity: {user_data['food_insecurity']:.0%}")
        risk_score += 0.25
    elif user_data['food_insecurity'] > 0.3:
        risk_factors.append(f"Moderate food insecurity: {user_data['food_insecurity']:.0%}")
        risk_score += 0.15

    # WASH conditions (weight: 20%)
    wash_score = user_data['clean_water_access'] + user_data['improved_sanitation']
    if wash_score == 0:
        risk_factors.append("Critical WASH: no clean water or sanitation")
        risk_score += 0.20
    elif wash_score == 1:
        risk_factors.append("Poor WASH: limited water/sanitation access")
        risk_score += 0.10

    # Health factors (weight: 20%)
    health_risks = 0
    if user_data['diarrhea_last_week'] == 1:
        health_risks += 1
        risk_factors.append("Recent diarrhea infection")
    if user_data['vaccination_complete'] == 0:
        health_risks += 1
        risk_factors.append("Incomplete vaccination")
    if user_data['breastfeeding'] == 0 and user_data['age_months'] < 24:
        health_risks += 1
        risk_factors.append("No breastfeeding (for child < 2 years)")

    risk_score += (health_risks / 3) * 0.20

    # Clinical assessment (weight: 10%)
    if user_data['stunting_risk_score'] > 35:
        risk_factors.append(f"High clinical risk: {user_data['stunting_risk_score']}%")
        risk_score += 0.10

    return risk_factors, min(risk_score, 1.0)  # Cap at 1.0

def encode_categorical_features(input_data: PredictionInput) -> Dict[str, Any]:
    """
    Encode categorical string features to numerical values
    """
    encoded_data = {}
    
    try:
        # Encode categorical variables
        encoded_data['rural_urban_encoded'] = label_encoders['rural_urban'].transform([input_data.rural_urban])[0]
        encoded_data['region_encoded'] = label_encoders['region'].transform([input_data.region])[0]
        encoded_data['mother_education_encoded'] = label_encoders['mother_education'].transform([input_data.mother_education])[0]
        
        print(f"Encoded categorical features:")
        print(f"  rural_urban: {input_data.rural_urban} -> {encoded_data['rural_urban_encoded']}")
        print(f"  region: {input_data.region} -> {encoded_data['region_encoded']}")
        print(f"  mother_education: {input_data.mother_education} -> {encoded_data['mother_education_encoded']}")
        
    except ValueError as e:
        print(f"Warning: Categorical value not in expected categories: {e}")
        # Use default values if unknown categories
        encoded_data['rural_urban_encoded'] = 0
        encoded_data['region_encoded'] = 0
        encoded_data['mother_education_encoded'] = 0
    
    return encoded_data

def preprocess_input_data(input_data: PredictionInput) -> pd.DataFrame:
    """
    Preprocess the input data to match the model's expected format with EXACTLY 15 features
    """
    # Encode categorical features first
    encoded_data = encode_categorical_features(input_data)
    
    # Calculate derived features
    wash_score = input_data.clean_water_access + input_data.improved_sanitation
    health_vulnerability = (
        input_data.diarrhea_last_week +
        (1 - input_data.vaccination_complete) +
        (1 - input_data.breastfeeding)
    )
    composite_risk_score = (
        input_data.food_insecurity +
        (1 - input_data.clean_water_access) +
        (1 - input_data.improved_sanitation) +
        health_vulnerability / 3
    )
    
    # Create exactly 15 features that match what the model expects
    # Adjust this order based on how your actual model was trained
    features = {
        'feature_0': input_data.age_months,                    # age_months
        'feature_1': input_data.household_income,              # household_income
        'feature_2': input_data.family_size,                   # family_size
        'feature_3': input_data.food_insecurity,               # food_insecurity
        'feature_4': input_data.breastfeeding,                 # breastfeeding
        'feature_5': input_data.vaccination_complete,          # vaccination_complete
        'feature_6': input_data.diarrhea_last_week,            # diarrhea_last_week
        'feature_7': input_data.clean_water_access,            # clean_water_access
        'feature_8': input_data.improved_sanitation,           # improved_sanitation
        'feature_9': input_data.stunting_risk_score,           # stunting_risk_score
        'feature_10': wash_score,                              # wash_score
        'feature_11': health_vulnerability,                    # health_vulnerability
        'feature_12': composite_risk_score,                    # composite_risk_score
        'feature_13': encoded_data['rural_urban_encoded'],     # rural_urban (encoded)
        'feature_14': encoded_data['region_encoded']           # region (encoded)
        # Note: We're only using 2 of the 3 encoded categorical features to keep it at 15
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    print(f"Processed features for prediction: {len(features)} features")
    print(f"Feature names: {list(features.keys())}")
    
    return df

@app.get("/")
async def root():
    return {
        "message": "Malnutrition Risk Prediction API",
        "status": "Running",
        "model_loaded": model_loaded,
        "model_type": "Dummy Model" if "dummy" in str(model).lower() else "Trained Model",
        "expected_features": 15,
        "expected_categories": expected_categories,
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_malnutrition_risk(input_data: PredictionInput):
    """
    Predict malnutrition risk based on input parameters
    
    All string fields (rural_urban, region, mother_education) are preserved as strings in input
    and automatically encoded to numerical values for the model prediction.
    """
    
    if not model_loaded:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Using dummy model for testing."
        )
    
    try:
        # Convert input to dictionary for validation
        user_data_dict = input_data.dict()
        
        # Validate input
        input_warnings = validate_user_input(user_data_dict)
        
        # Calculate risk factors
        risk_factors, manual_risk_score = calculate_risk_factors(user_data_dict)
        
        # Preprocess input data (this now handles categorical encoding)
        processed_data = preprocess_input_data(input_data)
        
        # Verify feature count
        if processed_data.shape[1] != 15:
            print(f"Warning: Processed data has {processed_data.shape[1]} features, but model expects 15")
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Get the highest probability and corresponding class
        max_proba = np.max(prediction_proba[0])
        predicted_class = prediction[0]
        
        # Map prediction to risk category
        risk_categories = {
            0: "Low Risk",
            1: "Medium Risk", 
            2: "High Risk"
        }
        
        risk_category = risk_categories.get(predicted_class, "Unknown")
        
        # Calculate confidence
        confidence = max_proba * 100
        
        return PredictionResponse(
            risk_category=risk_category,
            risk_probability=max_proba,
            confidence=confidence,
            input_validation_notes=input_warnings,
            risk_factors=risk_factors,
            manual_risk_score=manual_risk_score,
            model_type="Dummy Model" if "dummy" in str(model).lower() else "Trained Model"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)