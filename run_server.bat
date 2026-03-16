@echo off
cd /d "%~dp0"
echo Starting TruthLens Backend Server using Virtual Environment...
call .venv\Scripts\activate.bat
uvicorn api.main:app --reload --port 8000
pause
