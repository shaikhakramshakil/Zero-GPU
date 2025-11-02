/**
 * Cloud LLM VS Code Extension
 * 
 * Provides AI-powered code completions and assistance
 * by connecting to a remote LLM server (running on Colab/Kaggle)
 */

const vscode = require('vscode');
const apiClient = require('./src/apiClient');
const completionProvider = require('./src/completionProvider');

let statusBar;
let globalContext;
let isConnected = false;

/**
 * Extension activation
 */
async function activate(context) {
    console.log('🚀 Cloud LLM extension activated');
    globalContext = context;

    // Create status bar
    createStatusBar();

    // Register commands
    registerCommands(context);

    // Initialize settings
    loadSettings();

    // Auto-test connection on startup
    setTimeout(() => testConnection(false), 2000);
}

/**
 * Create status bar item
 */
function createStatusBar() {
    statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBar.command = 'cloudllm.testConnection';
    updateStatusBar('Loading...');
    statusBar.show();
}

/**
 * Update status bar
 */
function updateStatusBar(text, tooltip = '') {
    if (statusBar) {
        statusBar.text = `$(cloud) ${text}`;
        statusBar.tooltip = tooltip || text;
    }
}

/**
 * Register all commands
 */
function registerCommands(context) {
    // Set API Key
    context.subscriptions.push(
        vscode.commands.registerCommand('cloudllm.setApiKey', setApiKey)
    );

    // Test Connection
    context.subscriptions.push(
        vscode.commands.registerCommand('cloudllm.testConnection', () => testConnection(true))
    );

    // Get Completion
    context.subscriptions.push(
        vscode.commands.registerCommand('cloudllm.getCompletion', getCompletion)
    );

    // Clear Cache
    context.subscriptions.push(
        vscode.commands.registerCommand('cloudllm.clearCache', clearCache)
    );

    // Register inline completion provider
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            completionProvider.createProvider(apiClient, getConfig)
        )
    );
}

/**
 * Set API Key and URL
 */
async function setApiKey() {
    try {
        // Get API URL
        const apiUrl = await vscode.window.showInputBox({
            prompt: 'Enter Cloud LLM API URL',
            placeholder: 'e.g., http://localhost:8000 or https://xxxxx.ngrok.io',
            value: getConfig().apiUrl || '',
            validateInput: (value) => {
                if (!value) return 'URL is required';
                try {
                    new URL(value);
                    return null;
                } catch {
                    return 'Invalid URL format';
                }
            }
        });

        if (!apiUrl) return;

        // Get API Key
        const apiKey = await vscode.window.showInputBox({
            prompt: 'Enter your Cloud LLM API Key',
            placeholder: 'Bearer token from Colab setup',
            password: true,
            value: getConfig().apiKey || '',
            validateInput: (value) => {
                if (!value) return 'API Key is required';
                if (value.length < 20) return 'API Key seems too short';
                return null;
            }
        });

        if (!apiKey) return;

        // Save settings
        const config = vscode.workspace.getConfiguration('cloudllm');
        await config.update('apiUrl', apiUrl, vscode.ConfigurationTarget.Global);
        await config.update('apiKey', apiKey, vscode.ConfigurationTarget.Global);

        updateStatusBar('$(cloud-download) Connecting...');

        // Test connection
        const result = await apiClient.testConnection(apiUrl, apiKey);
        
        if (result.success) {
            isConnected = true;
            updateStatusBar('$(cloud-check) Connected', `Connected to ${result.model}`);
            vscode.window.showInformationMessage(
                `✓ Successfully connected to Cloud LLM!\n\nModel: ${result.model}\n\nUse Ctrl+L to get completions`
            );
        } else {
            isConnected = false;
            updateStatusBar('$(cloud-error) Failed');
            vscode.window.showErrorMessage(`Failed to connect: ${result.error}`);
        }
    } catch (error) {
        updateStatusBar('$(cloud-error) Error');
        vscode.window.showErrorMessage(`Error setting API key: ${error.message}`);
    }
}

/**
 * Test connection to API
 */
async function testConnection(showSuccess = true) {
    try {
        const config = getConfig();
        
        if (!config.apiUrl || !config.apiKey) {
            updateStatusBar('$(cloud-error) Not configured');
            if (showSuccess) {
                vscode.window.showWarningMessage(
                    'Cloud LLM not configured. Click "Set API Key" to configure.',
                    { title: 'Configure' }
                ).then(item => {
                    if (item) {
                        vscode.commands.executeCommand('cloudllm.setApiKey');
                    }
                });
            }
            return;
        }

        updateStatusBar('$(sync) Testing...');

        const result = await apiClient.testConnection(config.apiUrl, config.apiKey);

        if (result.success) {
            isConnected = true;
            updateStatusBar('$(cloud-check) Connected', `Model: ${result.model}`);
            if (showSuccess) {
                vscode.window.showInformationMessage(
                    `✓ Connected to Cloud LLM\n\nModel: ${result.model}\n\nPress Ctrl+L for completions`
                );
            }
        } else {
            isConnected = false;
            updateStatusBar('$(cloud-error) Disconnected');
            if (showSuccess) {
                vscode.window.showErrorMessage(`Connection failed: ${result.error}`);
            }
        }
    } catch (error) {
        isConnected = false;
        updateStatusBar('$(cloud-error) Error');
        if (showSuccess) {
            vscode.window.showErrorMessage(`Test failed: ${error.message}`);
        }
    }
}

/**
 * Get completion for current code
 */
async function getCompletion() {
    try {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        if (!isConnected) {
            vscode.window.showWarningMessage('Not connected to Cloud LLM. Configure API key first.');
            return;
        }

        const config = getConfig();
        const document = editor.document;
        const position = editor.selection.active;
        const text = document.getText(new vscode.Range(new vscode.Position(0, 0), position));

        updateStatusBar('$(loading~spin) Getting completion...');

        const response = await apiClient.getCompletion(
            config.apiUrl,
            config.apiKey,
            text,
            config.maxTokens,
            config.temperature
        );

        if (response.success) {
            // Insert completion
            await editor.edit(editBuilder => {
                editBuilder.insert(position, response.completion);
            });

            updateStatusBar('$(cloud-check) Completion inserted');
            
            setTimeout(() => {
                updateStatusBar('$(cloud-check) Connected');
            }, 2000);
        } else {
            updateStatusBar('$(cloud-error) Failed');
            vscode.window.showErrorMessage(`Completion failed: ${response.error}`);
        }
    } catch (error) {
        updateStatusBar('$(cloud-error) Error');
        vscode.window.showErrorMessage(`Error getting completion: ${error.message}`);
    }
}

/**
 * Clear all cached settings
 */
async function clearCache() {
    try {
        const config = vscode.workspace.getConfiguration('cloudllm');
        await config.update('apiUrl', '', vscode.ConfigurationTarget.Global);
        await config.update('apiKey', '', vscode.ConfigurationTarget.Global);
        
        isConnected = false;
        updateStatusBar('$(cloud-error) Not configured');
        vscode.window.showInformationMessage('Cache cleared. Please reconfigure API key.');
    } catch (error) {
        vscode.window.showErrorMessage(`Error clearing cache: ${error.message}`);
    }
}

/**
 * Load configuration
 */
function getConfig() {
    const config = vscode.workspace.getConfiguration('cloudllm');
    return {
        apiUrl: config.get('apiUrl', ''),
        apiKey: config.get('apiKey', ''),
        maxTokens: config.get('maxTokens', 512),
        temperature: config.get('temperature', 0.7),
        enableAutoComplete: config.get('enableAutoComplete', true),
        showStatusBar: config.get('showStatusBar', true)
    };
}

/**
 * Load settings on startup
 */
function loadSettings() {
    const config = getConfig();
    if (!config.showStatusBar) {
        statusBar.hide();
    }
}

/**
 * Extension deactivation
 */
function deactivate() {
    console.log('Cloud LLM extension deactivated');
    if (statusBar) {
        statusBar.dispose();
    }
}

module.exports = {
    activate,
    deactivate
};
