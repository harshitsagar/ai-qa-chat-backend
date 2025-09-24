import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # Model configuration - using Groq's fast models
    MODEL_NAME = "llama3-8b-8192"  # You can also use "mixtral-8x7b-32768" or "gemma-7b-it"

settings = Settings()