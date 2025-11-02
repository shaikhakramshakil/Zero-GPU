/**
 * API Client for Cloud LLM
 * Handles communication with the remote LLM server
 */

const https = require('https');
const http = require('http');

/**
 * Make HTTP request to API
 */
function makeRequest(url, method, headers, body = null) {
    return new Promise((resolve, reject) => {
        try {
            const isHttps = url.startsWith('https');
            const client = isHttps ? https : http;
            const urlObj = new URL(url);

            const options = {
                hostname: urlObj.hostname,
                port: urlObj.port,
                path: urlObj.pathname + urlObj.search,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    ...headers
                },
                timeout: 30000
            };

            const req = client.request(options, (res) => {
                let data = '';

                res.on('data', (chunk) => {
                    data += chunk;
                });

                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        resolve({
                            status: res.statusCode,
                            data: response,
                            headers: res.headers
                        });
                    } catch (e) {
                        resolve({
                            status: res.statusCode,
                            data: data,
                            headers: res.headers
                        });
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            if (body) {
                req.write(JSON.stringify(body));
            }

            req.end();
        } catch (error) {
            reject(error);
        }
    });
}

/**
 * Test connection to Cloud LLM API
 */
async function testConnection(apiUrl, apiKey) {
    try {
        if (!apiUrl || !apiKey) {
            return {
                success: false,
                error: 'API URL and Key are required'
            };
        }

        const healthUrl = `${apiUrl}/health`;
        const headers = {
            'Authorization': `Bearer ${apiKey}`
        };

        const response = await makeRequest(healthUrl, 'GET', headers);

        if (response.status === 200 && response.data.model) {
            return {
                success: true,
                model: response.data.model,
                status: response.data.status
            };
        } else if (response.status === 401) {
            return {
                success: false,
                error: 'Invalid API key'
            };
        } else {
            return {
                success: false,
                error: `Server returned status ${response.status}`
            };
        }
    } catch (error) {
        return {
            success: false,
            error: error.message || 'Connection failed'
        };
    }
}

/**
 * Get text completion from API
 */
async function getCompletion(apiUrl, apiKey, prompt, maxTokens = 512, temperature = 0.7) {
    try {
        if (!apiUrl || !apiKey) {
            return {
                success: false,
                error: 'API URL and Key are required'
            };
        }

        const completionUrl = `${apiUrl}/v1/completions`;
        const headers = {
            'Authorization': `Bearer ${apiKey}`
        };

        const body = {
            prompt: prompt,
            max_tokens: Math.min(maxTokens, 2048),
            temperature: temperature,
            top_p: 0.95
        };

        const response = await makeRequest(completionUrl, 'POST', headers, body);

        if (response.status === 200 && response.data.completion) {
            return {
                success: true,
                completion: response.data.completion,
                tokens_used: response.data.tokens_used
            };
        } else if (response.status === 401) {
            return {
                success: false,
                error: 'Invalid API key'
            };
        } else if (response.status === 400) {
            return {
                success: false,
                error: response.data.detail || 'Bad request'
            };
        } else {
            return {
                success: false,
                error: `Server error: ${response.data.detail || response.status}`
            };
        }
    } catch (error) {
        return {
            success: false,
            error: error.message || 'Completion request failed'
        };
    }
}

/**
 * Get chat completion (OpenAI compatible)
 */
async function getChatCompletion(apiUrl, apiKey, messages, maxTokens = 512, temperature = 0.7) {
    try {
        if (!apiUrl || !apiKey) {
            return {
                success: false,
                error: 'API URL and Key are required'
            };
        }

        const chatUrl = `${apiUrl}/v1/chat/completions`;
        const headers = {
            'Authorization': `Bearer ${apiKey}`
        };

        const body = {
            messages: messages,
            max_tokens: Math.min(maxTokens, 2048),
            temperature: temperature,
            top_p: 0.95
        };

        const response = await makeRequest(chatUrl, 'POST', headers, body);

        if (response.status === 200 && response.data.choices) {
            const completion = response.data.choices[0]?.message?.content || '';
            return {
                success: true,
                completion: completion,
                model: response.data.model
            };
        } else if (response.status === 401) {
            return {
                success: false,
                error: 'Invalid API key'
            };
        } else {
            return {
                success: false,
                error: response.data.detail || `Server error: ${response.status}`
            };
        }
    } catch (error) {
        return {
            success: false,
            error: error.message || 'Chat request failed'
        };
    }
}

module.exports = {
    testConnection,
    getCompletion,
    getChatCompletion,
    makeRequest
};
