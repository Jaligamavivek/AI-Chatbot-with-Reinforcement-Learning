"""
WSGI configuration for PythonAnywhere deployment
"""
import sys
import os

# Add your project directory to the sys.path
project_home = '/home/yourusername/AI-Chatbot-with-Reinforcement-Learning'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['FLASK_ENV'] = 'production'

# Import Flask app
from app import app as application
