import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    # Read secrets from environment variables (do NOT hard-code API keys)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Add other configurations like model names, file storage paths, etc.
    UPLOAD_FOLDER = "storage/uploaded_files"
    PROCESSED_DATA_FOLDER = "storage/processed_data"
    # Ensure these folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    # LLM Models
    GROQ_MODEL = "llama3-8b-8192" # Or "llama3-70b-8192"
    GEMINI_MODEL = "gemini-pro"