@echo off
echo Starting Flux LoRA Pipeline Frontend...
cd /d "%~dp0\..\frontend"
call npm run dev
pause
