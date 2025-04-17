from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from typing import List

app = FastAPI()

# Cargar modelo
model_data = joblib.load('model.joblib')
model = model_data['pipeline']
le = model_data['label_encoder']

# Cargar metadata
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

class StudentData(BaseModel):
    marital_status: int
    application_mode: int
    application_order: int
    daytime_evening_attendance: int
    previous_qualification: int
    previous_qualification_grade: float
    admission_grade: float
    debtor: bool
    tuition_fees_up_to_date: bool
    gender: int
    scholarship_holder: bool
    age_at_enrollment: int

@app.post("/predict")
def predict(student_data: StudentData):
    # Convertir a DataFrame
    input_data = {
        'Marital status': [student_data.marital_status],
        'Application mode': [student_data.application_mode],
        'Application order': [student_data.application_order],
        'Daytime/evening attendance': [student_data.daytime_evening_attendance],
        'Previous qualification': [student_data.previous_qualification],
        'Previous qualification (grade)': [student_data.previous_qualification_grade],
        'Admission grade': [student_data.admission_grade],
        'Debtor': [student_data.debtor],
        'Tuition fees up to date': [student_data.tuition_fees_up_to_date],
        'Gender': [student_data.gender],
        'Scholarship holder': [student_data.scholarship_holder],
        'Age at enrollment': [student_data.age_at_enrollment]
    }
    
    df = pd.DataFrame(input_data)
    
    # Predecir
    try:
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        
        # Convertir predicción a label original
        prediction_label = le.inverse_transform(prediction)[0]
        
        return {
            "prediction": int(prediction[0]),
            "prediction_label": str(prediction_label),
            "probability": float(prediction_proba[0][1]),
            "message": "Graduate" if prediction[0] == 1 else "Dropout"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/model_info")
def get_model_info():
    return metadata