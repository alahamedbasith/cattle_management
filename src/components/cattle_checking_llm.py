# app/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from src.utils import load_and_preprocess_image, predict_cattle_class, display_registration_details, get_gemini_response
from src.components.prediction import model, df_cattle, labels  # Import from models
import numpy as np
from PIL import Image
import io

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
    "2. Provide bounding box coordinates for the cow's muzzle in the format [ymin, xmin, ymax, xmax]. "
    "3. Focus specifically on the muzzle area, including the nose and mouth. The area must be clearly visible, in focus, and close-up to be classified as 'yes'. "
    "4. If the image is far from the camera, shows a farm or landscape, or does not have a clear cow muzzle, return 'no'. "
    "5. If the image is not suitable for cattle identification or does not show a cow's muzzle, suggest 'The Cattle Registration was not found'. "
    "6. Identify all object types present in the image, and consider the context (e.g., farm, landscape) when classifying. "
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
                        bounding_box = eval(bounding_box_str)  # Be cautious with eval in production code!
                        message_part = bounding_box_part[1].split(", Object Type:")
                        if len(message_part) > 1:
                            message_response = message_part[0].strip()
                            object_types = message_part[1].strip()
                            
                            # Split the object types into a list
                            object_type_list = [obj.strip() for obj in object_types.split(",")]

        # Convert "yes/no" classification to boolean
        muzzle_identified = classification_response.lower() == "yes"

        # If a cow's muzzle is identified, proceed with prediction
        if muzzle_identified:
            # Preprocess the image for prediction
            img_array = load_and_preprocess_image(np.array(pil_image))

            # Make predictions about the cattle class
            predicted_class, predicted_prob = predict_cattle_class(model, img_array)

            if predicted_class:
                registration_details = display_registration_details(predicted_class)  # Get registration details
                return {
                    "muzzle_identified": muzzle_identified,
                    "bounding_box": bounding_box,
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
            "bounding_box": bounding_box,
            "identified_objects": object_type_list,
            "message": "The Cattle Registration was not found",
            "cattle_info": {}
        }

    except Exception as e:
        # Log the error and raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
