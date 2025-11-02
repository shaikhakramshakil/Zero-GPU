#!/bin/bash
# filepath: setup-linux.sh

echo ""
echo "========================================="
echo "  Cloud LLM - Linux/Mac Setup"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Please install Python3 first"
    exit 1
fi

echo "[1/4] Installing Python dependencies..."
pip3 install -q vllm fastapi uvicorn pydantic requests pyngrok

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "[2/4] Verifying installation..."
python3 -c "import vllm; import fastapi; print('  OK')"

if [ $? -ne 0 ]; then
    echo "ERROR: Verification failed"
    exit 1
fi

echo "[3/4] Checking VS Code..."
if ! command -v code &> /dev/null; then
    echo "WARNING: VS Code not found"
    echo "Please install from: https://code.visualstudio.com"
fi

echo "[4/4] Setup complete!"
echo ""
echo "========================================="
echo "  Next Steps:"
echo "========================================="
echo ""
echo "1. Go to: https://www.kaggle.com/code"
echo "2. Create new notebook"
echo "3. Enable GPU (Settings > Accelerator > P100)"
echo "4. Paste the code from README.md"
echo "5. Run the cell"
echo "6. Copy the API Key"
echo "7. Run: python3 vscode-setup/auto-configure.py YOUR_API_KEY"
echo ""
echo "========================================="
echo ""
