import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "iris_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict(features):
    prediction = model.predict([features])
    return int(prediction[0])
