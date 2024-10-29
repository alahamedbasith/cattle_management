# app/models.py
import os
import pandas as pd
import json
import logging
from tensorflow.keras.models import load_model

# Define the data directory paths
DATA_DIR = os.path.join(os.getcwd(), 'data')  
MODELS_DIR = os.path.join(os.getcwd(), 'models')

# Load the registration data
registration_file = os.path.join(DATA_DIR, 'sample_registration.csv')
df_cattle = pd.read_csv(registration_file)

# Load class labels from a JSON file
class_labels_file = os.path.join(MODELS_DIR, 'class_labels_vgg.json')
with open(class_labels_file, 'r') as json_file:
    labels = json.load(json_file)

# Load the pre-trained model
model_file = os.path.join(MODELS_DIR, 'vgg_model.h5')
model = load_model(model_file)

print("Model and data loaded successfully.")
logging.info("Model and data loaded successfully.")
