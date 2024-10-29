# app/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from src.utils import load_and_preprocess_image, predict_cattle_class, display_registration_details, get_gemini_response
from src.components.prediction import model, df_cattle, labels  
import numpy as np
import logging
from PIL import Image
import io
import os

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Cattle Management</title>
        </head>
        <body>
            <h1>Cattle Management</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/jpeg, image/png" required>
                <button type="submit">Tell me about the image</button>
            </form>
        </body>
    </html>
    """

combined_prompt = (
    "Classify the uploaded image based on the following criteria: "
    "1. If the image contains a cow's muzzle and the muzzle is large enough to indicate a close-up (muzzle covers at least 15% of the image), return 'yes'. Otherwise, return 'no'. "
    "2. Focus specifically on the muzzle area, defined as the nose and mouth only. The eyes or other facial features should not be included in the bounding box. "
    "3. If the image is far from the camera, shows a farm or landscape, or does not have a clear cow muzzle, return 'no'. "
    "4. If the image is not suitable for cattle identification or does not show a cow's muzzle, suggest 'The Cattle Registration was not found'. "
    "5. Identify all object types present in the image, and consider the context (e.g., farm, landscape) when classifying. "
    "6. Identify the cow’s muzzle in the image, focusing on the nose and mouth. Avoid including any part of the eyes or forehead in the bounding box. "
    "   The bounding box should tightly enclose the nose and mouth area, with minimal background. "
    "   Return the coordinates of the cow's muzzle area in bounding box format as precisely as possible, ensuring it excludes the eyes and upper face. "
    
    "Format the response as follows: Classification: <yes/no>, Bounding Box: [ymin, xmin, ymax, xmax], Message: <Provide an explanation>, Object Type: <object1, object2, ...>."
)



@router.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        combined_response = get_gemini_response(combined_prompt, pil_image)

        # Initialize variables for responses
        classification_response = ""
        bounding_box = []
        message_response = ""
        object_type_list = []

        # Parse the combined response safely
        if "Classification:" in combined_response and "Bounding Box:" in combined_response and "Message:" in combined_response and "Object Type:" in combined_response:
            parts = combined_response.split("Classification:")
            if len(parts) > 1:
                classification_part = parts[1].split(", Bounding Box:")
                if len(classification_part) > 1:
                    classification_response = classification_part[0].strip()
                    bounding_box_part = classification_part[1].split(", Message:")
                    if len(bounding_box_part) > 1:
                        bounding_box_str = bounding_box_part[0].strip()
                        # Convert the bounding box string to a list of floats
                        bounding_box = list(map(int, eval(bounding_box_str)))  # [ymin, xmin, ymax, xmax]
                        message_part = bounding_box_part[1].split(", Object Type:")
                        if len(message_part) > 1:
                            message_response = message_part[0].strip()
                            object_types = message_part[1].strip()
                            
                            # Split the object types into a list
                            object_type_list = [obj.strip() for obj in object_types.split(",")]

        # Convert "yes/no" classification to boolean
        muzzle_identified = classification_response.lower() == "yes"
        
        logging.info("Classified Successfully, Getting coordinates for cropping")

        # If a cow's muzzle is identified, proceed with prediction
        if muzzle_identified:
            # After loading the image
            image_array = np.array(pil_image)  # Convert to NumPy array
            height, width, _ = image_array.shape  # Get height and width

            # Your existing bounding box code here
            ymin, xmin, ymax, xmax = bounding_box
            x1 = int(xmin / 1000 * width)
            y1 = int(ymin / 1000 * height)
            x2 = int(xmax / 1000 * width)
            y2 = int(ymax / 1000 * height)


            # Crop the image based on the bounding box coordinates
            cropped_image = pil_image.crop((x1, y1, x2, y2))
            
            # Save the cropped image to a specified location on your laptop
            save_path = os.path.join("data", "cropped_image.png")  # Change to your desired path
            cropped_image.save(save_path)
            
            logging.info("Cropping Successfull")


            # Preprocess the cropped image for prediction
            img_array = load_and_preprocess_image(np.array(cropped_image))

            # Make predictions about the cattle class
            predicted_class, predicted_prob = predict_cattle_class(model, img_array)
            logging.info("Prediction Successfull")

            if predicted_class:
                registration_details = display_registration_details(predicted_class)  # Get registration details
                return {
                    "muzzle_identified": muzzle_identified,
                    "identified_objects": object_type_list,
                    "message": "The Cattle Registration was found",
                    "cattle_info": {
                        "predicted_class": predicted_class,
                        "probability": predicted_prob,
                        "details": registration_details
                    }
                }

        # Return a response if no cattle was found
        return {
            "muzzle_identified": muzzle_identified,
            "identified_objects": object_type_list,
            "message": "The Cattle Registration was not found",
            "cattle_info": {}
        }

    except Exception as e:
        # Log the error and raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
