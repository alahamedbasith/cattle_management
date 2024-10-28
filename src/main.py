# src/main.py
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
