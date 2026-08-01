"""Root app.py entrypoint for AGRI-SENS-CORE-V1."""

import os
import sys

# Add repository root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
