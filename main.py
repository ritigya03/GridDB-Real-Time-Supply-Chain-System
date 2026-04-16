"""
Entry point for the Supply Chain AI system.
Run with: uvicorn main:app --reload
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from api.app import app
