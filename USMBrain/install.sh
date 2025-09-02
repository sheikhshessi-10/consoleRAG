#!/bin/bash

echo "========================================"
echo "USM Brain Installation Script"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "Python found!"
python3 --version
echo

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed"
    echo "Please install pip3 first"
    exit 1
fi

echo "Installing dependencies..."
echo "This may take a few minutes..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to install some dependencies"
    echo "Please check the error messages above"
    exit 1
fi

echo
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Set your OpenAI API key in config.py or create a .env file"
echo "2. Run: python3 test_installation.py"
echo "3. Run: python3 USMBrain.py"
echo
