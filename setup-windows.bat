@echo off
REM filepath: setup-windows.bat
echo.
echo =========================================
echo   Cloud LLM - Windows Setup
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python from: https://www.python.org
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/4] Installing Python dependencies...
pip install -q vllm fastapi uvicorn pydantic requests pyngrok

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [2/4] Verifying installation...
python -c "import vllm; import fastapi; print('  OK')"

if %errorlevel% neq 0 (
    echo ERROR: Verification failed
    pause
    exit /b 1
)

echo [3/4] Checking VS Code...
where code >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: VS Code not found in PATH
    echo Please install VS Code from: https://code.visualstudio.com
)

echo [4/4] Setup complete!
echo.
echo =========================================
echo   Next Steps:
echo =========================================
echo.
echo 1. Go to: https://www.kaggle.com/code
echo 2. Create new notebook
echo 3. Enable GPU (Settings ^> Accelerator ^> P100)
echo 4. Paste the code from README.md
echo 5. Run the cell
echo 6. Copy the API Key
echo 7. Run: python vscode-setup\auto-configure.py YOUR_API_KEY
echo.
echo =========================================
echo.
pause
