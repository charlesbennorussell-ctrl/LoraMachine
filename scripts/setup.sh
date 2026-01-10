#!/bin/bash
set -e

echo "========================================"
echo "Flux LoRA Pipeline Setup (Linux/Mac)"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.10+ first"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js 18+ first"
    exit 1
fi

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Creating directories..."
mkdir -p models/{flux,loras,refiners}
mkdir -p outputs/{generated,refined,liked}
mkdir -p training_data

echo
echo "========================================"
echo "Setting up Python Backend"
echo "========================================"
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing PyTorch with CUDA 12.1..."
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install -r requirements.txt

cd ..

echo
echo "========================================"
echo "Setting up React Frontend"
echo "========================================"
cd frontend

echo "Installing npm packages..."
npm install

cd ..

echo
echo "========================================"
echo "Downloading Real-ESRGAN Model"
echo "========================================"
cd scripts
python download_models.py
cd ..

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "IMPORTANT: Before running, you need to:"
echo "1. Accept the Flux license at: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "2. Login to HuggingFace: huggingface-cli login"
echo
echo "To start the application:"
echo "  Terminal 1: cd backend && source venv/bin/activate && python main.py"
echo "  Terminal 2: cd frontend && npm run dev"
echo
echo "Then open http://localhost:5173 in your browser"
