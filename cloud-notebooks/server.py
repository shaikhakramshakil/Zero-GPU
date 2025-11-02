"""
Cloud LLM API Server
====================

FastAPI server for vLLM inference with authentication.
Supports both completion and chat completion endpoints.

Run with: python server.py
"""

import os
import json
import secrets
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams

# ============================================================================
# Configuration
# ============================================================================

API_KEY = os.getenv("API_KEY", secrets.token_urlsafe(32))
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 300))

print(f"Initializing LLM server...")
print(f"Model: {MODEL_ID}")
print(f"API Key: {API_KEY[:10]}...")

# ============================================================================
# Initialize vLLM
# ============================================================================

try:
    llm = LLM(
        model=MODEL_ID,
        dtype="half",  # Use float16 for memory efficiency
        max_model_len=4096,
        gpu_memory_utilization=0.9
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# ============================================================================
# Pydantic Models
# ============================================================================

class CompletionRequest(BaseModel):
    """Request for text completion"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

class CompletionResponse(BaseModel):
    """Response for text completion"""
    id: str
    prompt: str
    completion: str
    tokens_used: int
    model: str
    timestamp: str

class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # "system", "user", "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    """Request for chat completion"""
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    api_version: str
    timestamp: str

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Cloud LLM API",
    version="1.0.0",
    description="LLM inference API server running on Google Colab"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use: Bearer YOUR_API_KEY"
        )
    
    token = authorization.replace("Bearer ", "", 1)
    if token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return token

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=MODEL_ID,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Generate text completion
    
    Example:
    ```
    POST /v1/completions
    Authorization: Bearer YOUR_API_KEY
    
    {
        "prompt": "def hello_world():",
        "max_tokens": 100,
        "temperature": 0.7
    }
    ```
    """
    try:
        # Verify API key
        verify_api_key(authorization)
        
        # Validate input
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        if request.max_tokens > MAX_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens cannot exceed {MAX_TOKENS}"
            )
        
        if request.max_tokens < 1:
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be at least 1"
            )
        
        # Generate completion
        sampling_params = SamplingParams(
            temperature=max(0, min(2, request.temperature)),
            top_p=max(0, min(1, request.top_p)),
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )
        
        outputs = llm.generate(
            request.prompt,
            sampling_params,
            use_tqdm=False
        )
        
        if not outputs or not outputs[0].outputs:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate completion"
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
        raise HTTPException(
            status_code=500,
            detail=f"Completion generation failed: {str(e)}"
        )

@app.post("/v1/chat/completions")
async def chat_completions(
    request: dict,
    authorization: Optional[str] = Header(None)
):
    """
    Chat completions endpoint (OpenAI compatible)
    
    Example:
    ```
    POST /v1/chat/completions
    Authorization: Bearer YOUR_API_KEY
    
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    ```
    """
    try:
        # Verify API key
        verify_api_key(authorization)
        
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(
                status_code=400,
                detail="Messages cannot be empty"
            )
        
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "
        
        # Generate response
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.95)
        
        if max_tokens > MAX_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens cannot exceed {MAX_TOKENS}"
            )
        
        sampling_params = SamplingParams(
            temperature=max(0, min(2, temperature)),
            top_p=max(0, min(1, top_p)),
            max_tokens=max_tokens
        )
        
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        
        if not outputs or not outputs[0].outputs:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )
        
        completion_text = outputs[0].outputs[0].text
        
        return {
            "id": secrets.token_hex(8),
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(completion_text.split()),
                "total_tokens": len(prompt.split()) + len(completion_text.split())
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat completion failed: {str(e)}"
        )

@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models"""
    verify_api_key(authorization)
    
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "cloud-llm"
            }
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "Cloud LLM API",
        "version": "1.0.0",
        "model": MODEL_ID,
        "endpoints": {
            "health": "/health",
            "completions": "/v1/completions",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "docs": "/docs"
        },
        "authentication": "Bearer YOUR_API_KEY in Authorization header"
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Cloud LLM API Server")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"API Key: {API_KEY[:15]}...")
    print(f"Max Tokens: {MAX_TOKENS}")
    print(f"Docs: http://localhost:8000/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
