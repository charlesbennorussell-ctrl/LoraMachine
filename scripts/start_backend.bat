@echo off
echo Starting Flux LoRA Pipeline Backend...
cd /d "%~dp0\..\backend"
call venv\Scripts\activate.bat
python main.py
pause
