#!/bin/bash
# Auto-configure VS Code for Cloud LLM
# Bash Script for Mac/Linux

# Get parameters or ask user
if [ -z "$1" ]; then
    echo ""
    echo "===================================="
    echo "VS Code Auto-Configuration Tool"
    echo "===================================="
    echo ""
    read -p "Enter your API Key: " API_KEY
    read -p "Enter Server URL (default: http://localhost:8000): " SERVER_URL
    
    if [ -z "$SERVER_URL" ]; then
        SERVER_URL="http://localhost:8000"
    fi
else
    API_KEY="$1"
    SERVER_URL="${2:-http://localhost:8000}"
fi

# VS Code settings location
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    VS_CODE_SETTINGS="$HOME/Library/Application Support/Code/User/settings.json"
else
    # Linux
    VS_CODE_SETTINGS="$HOME/.config/Code/User/settings.json"
fi

# Check if VS Code is installed
if [ ! -f "$VS_CODE_SETTINGS" ]; then
    echo "ERROR: VS Code not found at $VS_CODE_SETTINGS"
    echo "Please install VS Code first from https://code.visualstudio.com/"
    exit 1
fi

# Create backup
cp "$VS_CODE_SETTINGS" "$VS_CODE_SETTINGS.backup"

# Update settings using Python (compatible with JSON)
python3 << EOF
import json
import sys

settings_file = "$VS_CODE_SETTINGS"

# Load existing settings
try:
    with open(settings_file, 'r') as f:
        settings = json.load(f)
except:
    settings = {}

# Add Cloud LLM settings
settings['cloud-llm.apiUrl'] = "$SERVER_URL"
settings['cloud-llm.apiKey'] = "$API_KEY"
settings['cloud-llm.maxTokens'] = 512
settings['cloud-llm.temperature'] = 0.7
settings['cloud-llm.topP'] = 0.95
settings['cloud-llm.autoComplete'] = True
settings['editor.suggest.preview'] = True
settings['editor.inlineSuggest.enabled'] = True

# Save settings
with open(settings_file, 'w') as f:
    json.dump(settings, f, indent=2)

print("✅ Settings updated successfully!")
EOF

echo ""
echo "===================================="
echo "✅ VS Code Configuration Complete!"
echo "===================================="
echo ""
echo "Settings Applied:"
echo "  • API Key: ${API_KEY:0:8}..."
echo "  • Server URL: $SERVER_URL"
echo "  • Max Tokens: 512"
echo "  • Temperature: 0.7"
echo ""
echo "📝 Next Steps:"
echo "  1. Restart VS Code"
echo "  2. Press Ctrl+L to get a code completion"
echo "  3. Done!"
echo ""
echo "📌 Backup saved to: $VS_CODE_SETTINGS.backup"
echo ""
