import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('model/data/churn_pipeline.pkl')


# Pydantic orqali Telco ustunlarini API qabul qiladigan qilib mosladik
class UserInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResult(BaseModel):
    prediction: int
    churn: str


@app.get('/status')
def status():
    return 'I am ok'


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict/', response_model=PredictionResult)
def predict(data: UserInput):
    # Kiritilgan ma'lumotlarni DataFrame ga aylantiramiz (Sizning toza uslubingiz)
    df = pd.DataFrame([{
        'gender': data.gender,
        'SeniorCitizen': data.SeniorCitizen,
        'Partner': data.Partner,
        'Dependents': data.Dependents,
        'tenure': data.tenure,
        'PhoneService': data.PhoneService,
        'MultipleLines': data.MultipleLines,
        'InternetService': data.InternetService,
        'OnlineSecurity': data.OnlineSecurity,
        'OnlineBackup': data.OnlineBackup,
        'DeviceProtection': data.DeviceProtection,
        'TechSupport': data.TechSupport,
        'StreamingTV': data.StreamingTV,
        'StreamingMovies': data.StreamingMovies,
        'Contract': data.Contract,
        'PaperlessBilling': data.PaperlessBilling,
        'PaymentMethod': data.PaymentMethod,
        'MonthlyCharges': data.MonthlyCharges,
        'TotalCharges': data.TotalCharges
    }])

    result = model['model'].predict(df)[0]
    churn = 'Yes' if result == 1 else 'No'

    return PredictionResult(prediction=int(result), churn=churn)