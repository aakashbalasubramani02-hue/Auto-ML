# Hugging Face Spaces entry point.
# HF Spaces looks for app.py at the repo root when SDK is set to "streamlit".
# This file simply re-exports everything from the actual Streamlit app.
from app.streamlit_app import *
