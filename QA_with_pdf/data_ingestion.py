import sys
import os
from llama_index.core import SimpleDirectoryReader
from errorhandling import CustomException
from logger import logger

from llama_index.readers.file import PyMuPDFReader

def load_data(data_directory):
    """
    Load data from a directory using SimpleDirectoryReader.
    """
    try:
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Directory not found: {data_directory}")
            
        logger.info("Data loading started...")
        # Force PyMuPDFReader for robust PDF parsing
        pdf_reader = PyMuPDFReader()
        reader = SimpleDirectoryReader(
            data_directory,
            file_extractor={".pdf": pdf_reader}
        )
        documents = reader.load_data()
        logger.info("Data loading completed successfully!")
        return documents
    except Exception as e:
        logger.error("Exception in Data Ingestion")
        raise CustomException(e, sys)
