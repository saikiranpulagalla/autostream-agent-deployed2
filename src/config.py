import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate that the API key is set
if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ GOOGLE_API_KEY not found. Please set it in .env file. "
        "Copy .env.example to .env and add your Google API key."
    )

# Set the API key in environment for LangChain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
