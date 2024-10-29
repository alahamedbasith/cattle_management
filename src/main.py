"""
import sys
from src.exception_handler import CustomException
from src.logger import logging

if __name__ == "__main__":
    
    logging.info("Started the task")

    try:
        a = 10 / 0 
    except Exception as e:
        logging.info("Errofiner occured")
        raise CustomException(e,sys)
"""

# src/main.py
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.components.cattle_checking_llm import router
from dotenv import load_dotenv
import logging

load_dotenv()  # Load environment variables
app = FastAPI()

# Include the router from the API module
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    logging.info("Started the task")
    uvicorn.run(app, host="127.0.0.1", port=8000)
