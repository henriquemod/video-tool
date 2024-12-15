from src.app import run_app
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


if __name__ == "__main__":
    run_app()
