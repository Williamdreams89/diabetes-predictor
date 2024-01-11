import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

print(BASE_DIR)

with open(f"{BASE_DIR}/diabetes_pred_model-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


features = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass',
       'pedigree', 'age', 'diabetes']

def prediction(input_data):
    scaler = StandardScaler()
    # Convert the input data as np array
    input_data_array = np.asarray(input_data)

    # Reshape the data since we are only predicting for a single instance
    reshaped_input_data = input_data_array.reshape(1, -1)

    # transform data into std data
    std_data = scaler.transform(reshaped_input_data)

    # Now make the prediction
    return model.predict(std_data)
