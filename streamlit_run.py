import subprocess
import os
import sys
from src.config import GOOGLE_API_KEY  # Load API key from .env

def run_streamlit():
    """Launches Streamlit with auto-reload."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port=8501"])

if __name__ == "__main__":
    run_streamlit()