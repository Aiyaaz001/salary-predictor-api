from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the FastAPI Application
app = FastAPI(title="Tech Salary Predictor API")

# 2. Load the deployment artifacts back into memory once when the server starts
model = joblib.load('deployment_artifacts/final_gradient_boosting_model.joblib')
scaler = joblib.load('deployment_artifacts/production_scaler.joblib')
model_features = joblib.load('deployment_artifacts/model_feature_columns.joblib')

# 3. Define the blueprint of incoming data using Pydantic
class JobInput(BaseModel):
    Rating: float
    company_age: float
    Industry: str
    Sector: str
    Size: str

@app.get("/")
def home():
    return {"message": "Salary Prediction API is Live and Operational!"}

@app.post("/predict")
def predict_salary(data: JobInput):
    # Convert incoming data into a dictionary using modern Pydantic v2 syntax
    raw_data = data.model_dump()
    
    # Initialize a clean single-row DataFrame matching training layout
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Map raw numeric parameters
    input_df['Rating'] = raw_data['Rating']
    input_df['company_age'] = raw_data['company_age']
    
    # Map categorical dummy variables dynamically
    for feature_name in ['Industry', 'Sector', 'Size']:
        dummy_col = f"{feature_name}_{raw_data[feature_name]}"
        if dummy_col in input_df.columns:
            input_df[dummy_col] = 1
            
    # Apply your loaded scaler transformations
    input_df[['Rating', 'company_age']] = scaler.transform(input_df[['Rating', 'company_age']])
    
    # Generate the prediction inference
    prediction = model.predict(input_df)[0]
    
    return {
        "status": "success",
        "predicted_average_salary_thousands": round(float(prediction), 2),
        "formatted_salary": f"${prediction:.2f}K per year"
    }
