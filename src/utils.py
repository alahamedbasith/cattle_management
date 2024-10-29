# app/utils.py
import cv2
import numpy as np
import os
from src.components.prediction import labels,df_cattle

def load_and_preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_cattle_class(model, img_array, threshold=0.55):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_prob = float(predictions[0][predicted_class_index])

    if predicted_class_prob > threshold:
        predicted_class = labels[str(predicted_class_index)]
        return predicted_class, predicted_class_prob
    else:
        return None, predicted_class_prob

def display_registration_details(predicted_class):
    if predicted_class:
        registration_details = df_cattle[df_cattle['Class'] == predicted_class]
        if not registration_details.empty:
            details_list = []
            for _, row in registration_details.iterrows():
                details = {
                    "Cattle ID": row['Cattle ID'],
                    "Cattle Breed": row['Breed'],
                    "Cattle Age Average": row['Age (Years)'],
                    "Owner Name": row['Owner Name'],
                    "Owner Contact": row['Owner Contact'],
                    "Registration Date": row['Registration Date']
                }
                details_list.append(details)
            return details_list
    return []

def get_gemini_response(input_text, image):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image])
    return response.text
