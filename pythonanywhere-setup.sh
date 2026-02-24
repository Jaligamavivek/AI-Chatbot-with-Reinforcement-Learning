#!/bin/bash
# PythonAnywhere Setup Script

echo "Installing dependencies..."
pip3 install --user -r requirements-local.txt

echo "Creating data directory..."
mkdir -p data

echo "Initializing database..."
python3 init_db.py

echo "Setup complete!"
echo "Now configure your web app in PythonAnywhere dashboard"
