import os
import sys
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from errorhandling import CustomException
from logger import logger

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_model():
    """
    Loads the Gemini Pro model.
    """
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            
        logger.info("Loading Gemini Pro Model...")
        model = Gemini(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error("Error loading model")
        raise CustomException(e, sys)
