# app/config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API")

logging.info("Obtain Gemini Key Successful")
