import os
import sys
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding
from errorhandling import CustomException
from logger import logger

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_embedding():
    """
    Loads the Gemini Embedding model.
    """
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        logger.info("Loading Gemini Embedding Model...")
        embedding_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)
        logger.info("Embedding Model loaded successfully!")
        return embedding_model
    except Exception as e:
        logger.error("Error loading embedding model")
        raise CustomException(e, sys)
