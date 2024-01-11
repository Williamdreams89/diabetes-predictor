from pydantic import BaseModel
from fastapi import FastAPI


import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

print(BASE_DIR)

with open(f"{BASE_DIR}/diabetes_pred_model-{__version__}.sav", "rb") as f:
    model = pickle.load(f)


app = FastAPI()


class ModelInputParams(BaseModel):
    pregnant : int
    glucose : int
    pressure : int
    triceps : int
    insulin : int
    mass: float
    pedigree: float
    age: int


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": __version__}


@app.post("/predict" )
def predict(input_parameters: ModelInputParams):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)
    pregnant = input_dict['pregnant']
    glucose = input_dict['glucose']
    pressure =input_dict['pressure']
    triceps = input_dict['triceps']
    insulin = input_dict['insulin']
    mass = input_dict['mass']
    pedigree = input_dict['pedigree']
    age = input_dict['age']
    
    input_list = (pregnant, glucose, pressure, triceps, insulin, mass, pedigree, age)
    input_data_as_array = np.asarray(input_list)
    reshaped_input_data = input_data_as_array.reshape(1, -1)
    std_data = StandardScaler().fit_transform(reshaped_input_data)
    prediction_outcome = model.predict(std_data)

    if prediction_outcome[0]==0:
        return 'The person is not diabetic'
    return 'The person is diabetic'