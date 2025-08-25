from fastapi import FastAPI
from app.schemas import IrisFeatures
from app.model import predict
from sklearn.datasets import load_iris

app = FastAPI()
iris = load_iris()

@app.get("/")
def read_root():
    return {"message": "Iris Prediction API is running."}

@app.post("/predict")
def get_prediction(data: IrisFeatures):
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]
    prediction = predict(features)
    return {
        "prediction": iris.target_names[prediction]
    }
