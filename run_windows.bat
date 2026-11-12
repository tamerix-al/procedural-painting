@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Install Python first.
        pause
        exit /b 1
    )
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
