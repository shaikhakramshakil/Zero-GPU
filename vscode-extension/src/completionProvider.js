/**
 * Inline Completion Provider for Cloud LLM
 * Provides real-time code completions as you type
 */

const vscode = require('vscode');

/**
 * Create inline completion provider
 */
function createProvider(apiClient, getConfig) {
    return {
        async provideInlineCompletionItems(document, position, context) {
            try {
                const config = getConfig();

                // Check if inline completion is enabled
                if (!config.enableAutoComplete) {
                    return [];
                }

                // Get the text up to current position
                const text = document.getText(
                    new vscode.Range(new vscode.Position(0, 0), position)
                );

                // Don't trigger on every keystroke - only on certain conditions
                if (!shouldTriggerCompletion(context, text)) {
                    return [];
                }

                // Don't trigger on very short prompts
                if (text.trim().length < 5) {
                    return [];
                }

                // Get completion from API
                const response = await apiClient.getCompletion(
                    config.apiUrl,
                    config.apiKey,
                    text,
                    config.maxTokens,
                    config.temperature
                );

                if (!response.success) {
                    return [];
                }

                // Clean up completion text
                const completion = response.completion.trim();

                if (!completion) {
                    return [];
                }

                // Create inline completion item
                const item = new vscode.InlineCompletionItem(
                    completion,
                    new vscode.Range(position, position)
                );

                return [item];

            } catch (error) {
                console.error('Completion error:', error);
                return [];
            }
        }
    };
}

/**
 * Determine if completion should be triggered
 */
function shouldTriggerCompletion(context, text) {
    // Trigger on specific characters
    const triggerChars = ['\n', '{', '(', '[', ' '];
    
    if (context.triggerKind === vscode.InlineCompletionTriggerKind.Automatic) {
        const lastChar = text[text.length - 1];
        return triggerChars.includes(lastChar);
    }

    // Always trigger on manual request
    return context.triggerKind === vscode.InlineCompletionTriggerKind.Explicit;
}

/**
 * Extract context around current position
 */
function getContextWindow(document, position, windowSize = 500) {
    const text = document.getText();
    const offset = document.offsetAt(position);
    
    const start = Math.max(0, offset - windowSize);
    const end = Math.min(text.length, offset + windowSize);
    
    return text.substring(start, end);
}

/**
 * Get language-specific prompt formatting
 */
function formatPrompt(text, language) {
    const formatters = {
        'python': formatPythonPrompt,
        'javascript': formatJsPrompt,
        'typescript': formatJsPrompt,
        'java': formatJavaPrompt,
        'cpp': formatCppPrompt,
        'c': formatCPrompt
    };

    const formatter = formatters[language] || ((text) => text);
    return formatter(text);
}

function formatPythonPrompt(text) {
    // Add Python-specific context
    return `# Python code\n${text}`;
}

function formatJsPrompt(text) {
    // Add JS-specific context
    return `// JavaScript\n${text}`;
}

function formatJavaPrompt(text) {
    return `// Java\n${text}`;
}

function formatCppPrompt(text) {
    return `// C++\n${text}`;
}

function formatCPrompt(text) {
    return `// C\n${text}`;
}

module.exports = {
    createProvider,
    shouldTriggerCompletion,
    getContextWindow,
    formatPrompt
};
