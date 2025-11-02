# Auto-configure VS Code for Cloud LLM
# PowerShell Script for Windows

param(
    [string]$ApiKey = "",
    [string]$ServerUrl = "http://localhost:8000"
)

# Show menu if no parameters
if ($ApiKey -eq "") {
    Write-Host ""
    Write-Host "====================================" -ForegroundColor Cyan
    Write-Host "VS Code Auto-Configuration Tool" -ForegroundColor Green
    Write-Host "====================================" -ForegroundColor Cyan
    Write-Host ""
    
    $ApiKey = Read-Host "Enter your API Key"
    $ServerUrl = Read-Host "Enter Server URL (default: http://localhost:8000)"
    
    if ($ServerUrl -eq "") {
        $ServerUrl = "http://localhost:8000"
    }
}

# VS Code settings location
$VsCodeSettingsPath = "$env:APPDATA\Code\User\settings.json"

# Check if VS Code is installed
if (-not (Test-Path $VsCodeSettingsPath)) {
    Write-Host "ERROR: VS Code not found at $VsCodeSettingsPath" -ForegroundColor Red
    Write-Host "Please install VS Code first from https://code.visualstudio.com/" -ForegroundColor Yellow
    exit 1
}

# Load existing settings
$settings = @{}
if (Test-Path $VsCodeSettingsPath) {
    $settings = Get-Content $VsCodeSettingsPath | ConvertFrom-Json | ConvertTo-Hashtable
}

# Add Cloud LLM settings
$settings['cloud-llm.apiUrl'] = $ServerUrl
$settings['cloud-llm.apiKey'] = $ApiKey
$settings['cloud-llm.maxTokens'] = 512
$settings['cloud-llm.temperature'] = 0.7
$settings['cloud-llm.topP'] = 0.95
$settings['cloud-llm.autoComplete'] = $true
$settings['editor.suggest.preview'] = $true
$settings['editor.inlineSuggest.enabled'] = $true

# Save settings
$settings | ConvertTo-Json | Set-Content $VsCodeSettingsPath

Write-Host ""
Write-Host "✅ VS Code Configuration Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Settings Applied:" -ForegroundColor Yellow
Write-Host "  • API Key: $($ApiKey.Substring(0, 8))..." -ForegroundColor Cyan
Write-Host "  • Server URL: $ServerUrl" -ForegroundColor Cyan
Write-Host "  • Max Tokens: 512" -ForegroundColor Cyan
Write-Host "  • Temperature: 0.7" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Restart VS Code" -ForegroundColor Cyan
Write-Host "  2. Press Ctrl+L to get a code completion" -ForegroundColor Cyan
Write-Host "  3. Done!" -ForegroundColor Cyan
Write-Host ""

# Helper function to convert PSObject to Hashtable
function ConvertTo-Hashtable {
    param(
        [Parameter(ValueFromPipeline = $true)]
        [PSObject]$InputObject
    )
    
    if ($null -eq $InputObject) { return $null }
    
    if ($InputObject -is [hashtable]) { return $InputObject }
    
    $output = @{}
    $InputObject.PSObject.Properties | ForEach-Object {
        if ($_.Value -is [PSObject] -and $_.Value -isnot [string]) {
            $output[$_.Name] = ConvertTo-Hashtable($_.Value)
        } else {
            $output[$_.Name] = $_.Value
        }
    }
    return $output
}
