#!/bin/bash

echo "🚀 Starting Brain Tumor Segmentation System..."

# Check if the virtual environment is working properly
if [ -d "venv" ]; then
    # Try to run python to see if the venv absolute paths are still valid
    if ! ./venv/bin/python3 -c "import sys" &> /dev/null; then
        echo "⚠️ Virtual environment seems broken (did you move the folder?). Rebuilding..."
        rm -rf venv
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    echo "⬇️ Installing dependencies (this might take a few minutes)..."
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Apply the starlette upgrade fix automatically
    pip install --upgrade starlette
else
    source venv/bin/activate
fi

# Run the application
echo "🌐 Launching Streamlit..."
streamlit run app.py
