"""
Cloud LLM API Server for Google Colab
=====================================

This script sets up a vLLM inference server in Google Colab with:
- FastAPI for REST endpoints
- Model loading (Mistral-7B or Qwen-7B)
- API key authentication
- Public URL via ngrok

Run this in Google Colab to start the server.
"""

import os
import sys
import subprocess
import secrets
import json
from datetime import datetime

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================

print_header("Step 1: Installing Dependencies")

print_info("Installing vLLM, FastAPI, and dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "vllm>=0.3.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0"
], check=True)

print_success("All dependencies installed!")

# ============================================================================
# STEP 2: Configure Model Selection
# ============================================================================

print_header("Step 2: Configure Model")

# Default to Mistral-7B (faster loading)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "Mistral-7B"

print_info(f"Using model: {MODEL_NAME}")
print_info(f"Model ID: {MODEL_ID}")
print_info(f"Expected VRAM: ~5GB")

# ============================================================================
# STEP 3: Generate API Key
# ============================================================================

print_header("Step 3: Generating API Key")

API_KEY = secrets.token_urlsafe(32)
print_success(f"API Key Generated: {API_KEY}")

# ============================================================================
# STEP 4: Create FastAPI Server
# ============================================================================

print_header("Step 4: Creating FastAPI Server")

server_code = '''
import os
import json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
import asyncio
import secrets

# ============ Configuration ============
API_KEY = "{api_key}"
MODEL_ID = "{model_id}"
MAX_TOKENS = 2048
REQUEST_TIMEOUT = 300

# ============ Initialize vLLM ============
print("Loading model:", MODEL_ID)
llm = LLM(model=MODEL_ID, dtype="half", max_model_len=4096)
print("Model loaded successfully!")

# ============ Pydantic Models ============
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

class CompletionResponse(BaseModel):
    id: str
    prompt: str
    completion: str
    tokens_used: int
    model: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model: str
    api_version: str

# ============ FastAPI App ============
app = FastAPI(title="Cloud LLM API", version="1.0.0")

def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token

@app.get("/health")
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=MODEL_ID,
        api_version="1.0.0"
    )

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    _: str = Header(None, alias="authorization")
):
    """Generate text completion"""
    try:
        # Verify API key
        verify_api_key(_)
        
        # Validate input
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.max_tokens > MAX_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens cannot exceed {{MAX_TOKENS}}"
            )
        
        # Generate completion
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        outputs = llm.generate(
            request.prompt,
            sampling_params,
            use_tqdm=False
        )
        
        completion_text = outputs[0].outputs[0].text
        tokens_used = len(outputs[0].outputs[0].token_ids)
        
        return CompletionResponse(
            id=secrets.token_hex(8),
            prompt=request.prompt,
            completion=completion_text,
            tokens_used=tokens_used,
            model=MODEL_ID,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(
    request: dict,
    _: str = Header(None, alias="authorization")
):
    """Chat completions endpoint (compatible with OpenAI format)"""
    try:
        # Verify API key
        verify_api_key(_)
        
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {{content}}\\n"
            elif role == "user":
                prompt += f"User: {{content}}\\n"
            elif role == "assistant":
                prompt += f"Assistant: {{content}}\\n"
        
        prompt += "Assistant: "
        
        # Generate response
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=min(max_tokens, MAX_TOKENS)
        )
        
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        completion_text = outputs[0].outputs[0].text
        
        return {{
            "id": secrets.token_hex(8),
            "object": "chat.completion",
            "created": datetime.now().isoformat(),
            "model": MODEL_ID,
            "choices": [{{
                "index": 0,
                "message": {{
                    "role": "assistant",
                    "content": completion_text
                }},
                "finish_reason": "stop"
            }}],
            "usage": {{
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(completion_text.split()),
                "total_tokens": len(prompt.split()) + len(completion_text.split())
            }}
        }}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''

# Write server code to file
with open("llm_server.py", "w") as f:
    f.write(server_code.format(api_key=API_KEY, model_id=MODEL_ID))

print_success("FastAPI server created!")

# ============================================================================
# STEP 5: Install and Configure ngrok
# ============================================================================

print_header("Step 5: Setting up ngrok for Public URL")

print_info("Installing ngrok...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)

print_info("To get a public URL, you need an ngrok token:")
print_info("1. Go to https://dashboard.ngrok.com/auth/your-authtoken")
print_info("2. Copy your auth token")
print_info("3. Paste it when prompted")

# Note: In actual Colab, users would paste their ngrok token here
# For this script, we'll just document it

# ============================================================================
# STEP 6: Display Configuration
# ============================================================================

print_header("Configuration Complete!")

config_info = {
    "model": MODEL_NAME,
    "model_id": MODEL_ID,
    "api_key": API_KEY,
    "server_port": 8000,
    "max_tokens": MAX_TOKENS,
    "timestamp": datetime.now().isoformat()
}

print(f"{Colors.BOLD}API Configuration:{Colors.END}")
print(f"  Model: {config_info['model']}")
print(f"  API Key: {config_info['api_key']}")
print(f"  Port: {config_info['server_port']}")
print(f"  Max Tokens: {config_info['max_tokens']}")

# ============================================================================
# STEP 7: Start Server with ngrok
# ============================================================================

print_header("Starting Server")

start_server_code = '''
import subprocess
import time
from pyngrok import ngrok

# Set ngrok auth token (you would paste this)
# ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")

# Start FastAPI server in background
import threading
import sys

def run_server():
    subprocess.run([sys.executable, "llm_server.py"], check=True)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(10)

print("\\n" + "="*60)
print("Server is running!")
print("="*60)

# Create ngrok tunnel
try:
    public_url = ngrok.connect(8000, "http")
    print(f"✓ Public URL: {public_url}")
except Exception as e:
    print(f"Note: ngrok tunnel failed. Local access available at:")
    print(f"  http://localhost:8000")
    print(f"Error: {e}")

# Keep server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\\nServer stopped.")
    ngrok.disconnect()
'''

print_info("To start the server, run this code in a new Colab cell:")
print(f"""
```python
# Step 1: Copy llm_server.py code (already created)
# Step 2: Uncomment and set your ngrok token below:
# ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")

{start_server_code}
```
""")

# ============================================================================
# Final Instructions
# ============================================================================

print_header("Setup Complete! 🎉")

print(f"""{Colors.BOLD}Next Steps:{Colors.END}

1. {Colors.YELLOW}Get ngrok token:{Colors.END}
   - Go to: https://dashboard.ngrok.com/auth/your-authtoken
   - Copy your auth token

2. {Colors.YELLOW}Start the server:{Colors.END}
   - Run the code in the next cell with your ngrok token
   - Server will start loading the model (~2-3 minutes)

3. {Colors.YELLOW}Save your credentials:{Colors.END}
   - Copy the API Key: {API_KEY}
   - Copy the Public URL from ngrok output

4. {Colors.YELLOW}Configure VS Code extension:{Colors.END}
   - Open VS Code
   - Install "Cloud LLM" extension
   - Press Ctrl+Shift+P → "Cloud LLM: Set API Key"
   - Paste API Key and URL

{Colors.BOLD}API Key:{Colors.END}
{API_KEY}

{Colors.BOLD}Model:{Colors.END}
{MODEL_NAME} ({MODEL_ID})

{Colors.BOLD}Health Check:{Colors.END}
Once server is running, visit: http://localhost:8000/health

{Colors.BOLD}API Docs:{Colors.END}
Once server is running, visit: http://localhost:8000/docs
""")

print_info("Save this notebook URL to keep the server running!")
print_info("Colab will keep the notebook active as long as your browser tab is open")
