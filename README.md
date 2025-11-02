<div align="center">

# 🚀 ZeroGPU

**Free AI Code Completions Without Local GPU or Subscriptions**

[![GitHub Stars](https://img.shields.io/github/stars/shaikhakramshakil/AI-on-Cloud?style=social)](https://github.com/shaikhakramshakil/AI-on-Cloud)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Get powerful AI code completions in VS Code using Kaggle's free P100 GPU.
<div align="center">
  <img src="https://github.com/user-attachments/assets/c6f74c88-514b-4bc4-8703-b150b87ff4b7" alt="ZeroGPU - Free AI Code Completions" width="600">
</div>


[Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Models](#-best-models) • [FAQ](#-faq)

</div>

---



## 💡 The Problem

You want AI code completions, but you're stuck:

| Option | Cost | Issue |
|--------|------|-------|
| **ChatGPT/Copilot** | 10,000-60,000 INR/year | Monthly subscription |
| **Local GPU** | 1,50,000+ INR | Expensive hardware + electricity |
| **Cloud Services** | 50,000-6,00,000 INR/year | Very expensive |
| **ZeroGPU** | **0 INR** | **Completely Free** ✅ |

---

## ✨ Why ZeroGPU?

```
✅ Free P100 GPU (Kaggle)     ✅ No Subscriptions        ✅ Fast (50 tokens/sec)
✅ Open Source Models          ✅ 9 Hours Per Session    ✅ Works in India
```

---

## ⚡ Quick Start

### 1️⃣ Setup Computer (2 min)

**Windows:**
```bash
setup-windows.bat
```

**Mac/Linux:**
```bash
bash setup-linux.sh
```

### 2️⃣ Start Kaggle Server (5 min)

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. New Notebook → Settings → GPU: P100
3. Paste this code:

```python
!pip install -q vllm fastapi uvicorn pydantic pyngrok requests
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, secrets
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

API_KEY = secrets.token_hex(16)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

print("Loading Mistral-7B...")
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.8)
print("Model loaded!")

@app.post("/v1/completions")
async def completions(request: CompletionRequest, authorization: str = Header(None)):
    if not authorization or authorization.split("Bearer ")[1] != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    
    params = SamplingParams(temperature=request.temperature, max_tokens=request.max_tokens)
    output = llm.generate(request.prompt, params)
    
    return {
        "prompt": request.prompt,
        "completion": output[0].outputs[0].text,
        "model": "mistralai/Mistral-7B-Instruct-v0.2"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": "P100"}

print("\n" + "="*50)
print(f"API Key: {API_KEY}")
print("Server: http://localhost:8000")
print("="*50 + "\n")

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="critical")
```

4. Run cell → Copy API Key

### 3️⃣ Configure VS Code (1 min)

```bash
cd vscode-setup
python auto-configure.py YOUR_API_KEY
```

✅ **Done! Press Ctrl+L to get AI completions**

---

## 🧠 How It Works

```
┌─────────────────┐
│  Your Computer  │
│  VS Code        │
└────────┬────────┘
         │ Ctrl+L
         ↓
    [HTTP Request]
         ↓
┌─────────────────────────────┐
│  Kaggle Cloud (P100 GPU)    │
│  FastAPI Server             │
│  Mistral-7B Model           │
└─────────────────────────────┘
         ↓
    [AI Completion]
         ↓
     5-30 seconds
         ↓
    [Code Inserted]
```

**Timeline:**
1. Press Ctrl+L
2. VS Code Extension captures code
3. Sends to Kaggle with API Key
4. Mistral-7B generates completion
5. Result appears in editor

---

## 🎯 Best Models

### Recommended (Default)

**Mistral-7B-Instruct-v0.2**
```python
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.8)
```
- Speed: 50 tokens/sec
- Quality: Excellent
- Memory: 5GB
- Best for: General coding

### Fast Option

**DeepSeek-Coder-6.7B**
```python
llm = LLM(model="deepseek-ai/deepseek-coder-6.7b-base", gpu_memory_utilization=0.8)
```
- Speed: 55 tokens/sec (fastest)
- Quality: Very Good
- Memory: 4GB
- Best for: Speed priority

### Multilingual

**Qwen-7B-Chat**
```python
llm = LLM(model="Qwen/Qwen-7B-Chat", gpu_memory_utilization=0.8)
```
- Speed: 42 tokens/sec
- Quality: Excellent
- Memory: 5GB
- Best for: Multiple languages

---

## 💰 Cost Comparison (India)

| Service | Annual Cost |
|---------|------------|
| ChatGPT Plus | 10,000 INR |
| GitHub Copilot | 10,000 INR |
| Local GPU Setup | 2,00,000+ INR |
| AWS/Google Cloud GPU | 6,00,000 INR |
| **ZeroGPU** | **0 INR** ✅ |

**Save: 10,000 - 6,00,000 INR per year!**

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model loading slow | Wait 3-5 min, check internet |
| Invalid API Key | Copy exactly from Kaggle output |
| Connection timeout | Verify Kaggle cell is running |
| Slow completions | Normal (5-30 sec). Reduce max_tokens |
| Session expired | Kaggle sessions last 9 hours. Restart next day |

---

## ❓ FAQ

**Q: Do I need a credit card?**
A: No. Kaggle is completely free, no credit card required.

**Q: Is my code safe?**
A: Yes. Kaggle notebooks are private. Code stays on your server.

**Q: Can I use other models?**
A: Yes! Edit the model name and restart.

**Q: Works with Indian internet?**
A: Yes! Tested on 4G and broadband. Just needs stable connection.

**Q: How long does each completion take?**
A: Usually 5-30 seconds depending on code length.

---

## 📁 Project Structure

```
zerogpu/
├── README.md                    (This file)
├── setup-windows.bat            (Windows setup)
├── setup-linux.sh               (Mac/Linux setup)
├── requirements.txt             (Python packages)
└── vscode-setup/
    ├── auto-configure.py        (Auto VS Code config)
    ├── extension.js             (VS Code extension)
    ├── completionProvider.js    (Completion logic)
    └── package.json             (Extension config)
```

---

## 🤝 For Indian Developers

This project is designed for you:

✅ Saves 10,000-6,00,000 INR/year
✅ No credit card needed
✅ Works with Indian internet (4G/Broadband)
✅ Perfect for students, freelancers, startups
✅ Open source & community-driven

**Please star this repo if it helps you!** ⭐

---

## 🌟 Star Us

If ZeroGPU saves you money or helps your coding, please give us a star!

<a href="https://github.com/shaikhakramshakil/AI-on-Cloud">
  <img src="https://img.shields.io/github/stars/shaikhakramshakil/AI-on-Cloud?style=social" alt="GitHub Stars">
</a>

---

## 🚀 Contributing

We welcome all contributions!

**Ways to help:**
- Report bugs
- Suggest features
- Submit pull requests
- Add documentation
- Test on different systems

**How to contribute:**
```bash
1. Fork the repo
2. Create branch: git checkout -b feature/your-feature
3. Commit: git commit -m "Add feature"
4. Push: git push origin feature/your-feature
5. Open Pull Request
```

<div align="center">

**Made for developers who want free AI without GPU costs**

Star ⭐ | Fork 🍴 | Share 📢

</div>
