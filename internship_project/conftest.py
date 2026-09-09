"""conftest.py — Pytest configuration for the internship project."""
import sys
import os

# Ensure the project root is always on the path for test imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
