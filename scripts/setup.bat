@echo off
echo ========================================
echo Flux LoRA Pipeline Setup (Windows)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

echo Creating directories...
mkdir models\flux 2>nul
mkdir models\loras 2>nul
mkdir models\refiners 2>nul
mkdir outputs\generated 2>nul
mkdir outputs\refined 2>nul
mkdir outputs\liked 2>nul
mkdir training_data 2>nul

echo.
echo ========================================
echo Setting up Python Backend
echo ========================================
cd backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA 12.1...
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing other dependencies...
pip install -r requirements.txt

cd ..

echo.
echo ========================================
echo Setting up React Frontend
echo ========================================
cd frontend

echo Installing npm packages...
call npm install

cd ..

echo.
echo ========================================
echo Downloading Real-ESRGAN Model
echo ========================================
cd scripts
python download_models.py
cd ..

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo IMPORTANT: Before running, you need to:
echo 1. Accept the Flux license at: https://huggingface.co/black-forest-labs/FLUX.1-dev
echo 2. Login to HuggingFace: huggingface-cli login
echo.
echo To start the application:
echo   1. Open Terminal 1: cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo   2. Open Terminal 2: cd frontend ^&^& npm run dev
echo.
echo Then open http://localhost:5173 in your browser
echo.
pause
