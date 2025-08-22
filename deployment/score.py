import os
import joblib
import json
import numpy as np

# Called when the service is loaded
def init():
    global model
    # Get the path to the model file using the AZUREML_MODEL_DIR environment variable
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'heart_disease_model.pkl')
    # Load the model
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    try:
        # Get the input data from the request
        data = json.loads(raw_data)['data']
        data = np.array(data)
        
        # Make prediction
        result = model.predict(data)
        
        # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error