"""
Main application module
Entry point for the image processing application
"""
import sys
import os

# Add parent directory to sys.path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from gui.app import run

if __name__ == "__main__":
    run()
