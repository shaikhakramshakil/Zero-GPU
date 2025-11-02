"""
Auto-Configure VS Code Extension
Python script that patches VS Code settings automatically
"""

import json
import os
import sys
import platform
from pathlib import Path

def get_vscode_settings_path():
    """Get VS Code settings path based on OS"""
    system = platform.system()
    
    if system == "Windows":
        return Path(os.getenv('APPDATA')) / "Code" / "User" / "settings.json"
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    else:  # Linux
        return Path.home() / ".config" / "Code" / "User" / "settings.json"

def auto_configure(api_key, server_url="http://localhost:8000"):
    """Automatically configure VS Code settings"""
    
    settings_path = get_vscode_settings_path()
    
    print(f"\n{'='*50}")
    print("VS Code Auto-Configuration Tool")
    print(f"{'='*50}\n")
    
    # Check if settings file exists
    if not settings_path.exists():
        print(f"❌ ERROR: VS Code settings not found at:")
        print(f"   {settings_path}")
        print("\n   Please install VS Code first!")
        return False
    
    # Load existing settings
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except:
        settings = {}
    
    # Create backup
    backup_path = settings_path.with_stem(settings_path.stem + ".backup")
    with open(backup_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"📌 Backup saved to: {backup_path}")
    
    # Add Cloud LLM settings
    settings['cloud-llm.apiUrl'] = server_url
    settings['cloud-llm.apiKey'] = api_key
    settings['cloud-llm.maxTokens'] = 512
    settings['cloud-llm.temperature'] = 0.7
    settings['cloud-llm.topP'] = 0.95
    settings['cloud-llm.autoComplete'] = True
    settings['editor.suggest.preview'] = True
    settings['editor.inlineSuggest.enabled'] = True
    
    # Save updated settings
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"\n✅ Settings Updated: {settings_path}\n")
    print("Settings Applied:")
    print(f"  • API Key: {api_key[:8]}...")
    print(f"  • Server URL: {server_url}")
    print(f"  • Max Tokens: 512")
    print(f"  • Temperature: 0.7")
    print("\n📝 Next Steps:")
    print("  1. Restart VS Code")
    print("  2. Press Ctrl+L to get code completions")
    print("  3. Done!\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python auto-configure.py <API_KEY> [SERVER_URL]")
        print("Example: python auto-configure.py a1b2c3d4e5f6g7h8")
        print("         python auto-configure.py a1b2c3d4e5f6g7h8 http://localhost:8000\n")
        sys.exit(1)
    
    api_key = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    success = auto_configure(api_key, server_url)
    sys.exit(0 if success else 1)
