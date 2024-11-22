// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
const openai = require('openai');
const path = require('path');  // Add this line
require('dotenv').config({ path: path.join(__dirname, '.env') });

console.log('API Key exists:', !!process.env.OPENAI_API_KEY);

let currentPanel = undefined;
let conversationHistory = [];

// Add OpenAI client configuration
const openaiClient = new openai.OpenAI({
    apiKey: process.env.OPENAI_API_KEY || getOpenAIKey() 
});

// Debounce utility function
function debounce(func, delay) {
	let timeout;
	return (...args) => {
			clearTimeout(timeout);
			timeout = setTimeout(() => func(...args), delay);
	};
}

async function fetchPrediction(text) {
	const route = "https://fjru3p67wsonsdfbdp7p673giq0cvtep.lambda-url.us-east-2.on.aws";
	if (text.trim() !== '') {
		const body = text;

		try {
			const response = await fetch(route, {
				method: 'POST',
				body: body,
				headers: { 'Content-Type': 'text/plain' },
			});
			
			if (response.ok) {
				const res = await response.text();
				console.log("Res: ", res)
				return res;
			} else {
				console.error('Error:', response.status);
				return "";
			}
		} catch (error) {
			console.error('Fetch error:', error);
			return "";
		}
	} else {
		console.log("empty")
		return "";
	}
}

async function fetchChatResponse(question) {
    try {
        if (conversationHistory.length === 0) {
            conversationHistory.push({
                role: "system",
                content: "You are a helpful coding assistant. When answering follow-up questions, maintain context from the previous conversation and understand that 'What about X' usually means to apply the previous concept to X."
            });
        }

		conversationHistory.push({ role: "user", content: question });
		console.log("Current conversation:", conversationHistory);
		
        const completion = await openaiClient.chat.completions.create({
            model: "gpt-4o-mini",
            messages: conversationHistory,
            temperature: 0.9,
			max_tokens: 500,
			presence_penalty: 0.6,
			frequency_penalty: 0.6
        });

        const response =  completion.choices[0].message.content;
		conversationHistory.push({ role: "assistant", content: response });

		console.log("OpenAI response received");
		return response;
    } catch (error) {
        console.error('OpenAI API error:', error);
        return "Sorry, I encountered an error processing your request.";
    }
}

function getOpenAIKey() {
	if (process.env.OPENAI_API_KEY) {
		return process.env.OPENAI_API_KEY;
	}
    const config = vscode.workspace.getConfiguration('miniCopilot');
    return config.get('openAIKey');
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */

function activate(context) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "tutorial" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with  registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('mini-copilot.helloWorld', function () {
		console.log("activated!");
		vscode.window.showInformationMessage('Hello World from mini-copilot!');
	});

	const debouncedFetchSuggestions = debounce(async (document, position, token, resolve) => {
		console.log("triggered")
		const text = document.getText(new vscode.Range(new vscode.Position(Math.max(0, position.line - 5), 0), position));
		const route = "https://fjru3p67wsonsdfbdp7p673giq0cvtep.lambda-url.us-east-2.on.aws";
		const res = await fetchPrediction(text)
		if (res.length == 0) {
			resolve({ items: [] });
		} else {
			const completionItem = new vscode.InlineCompletionItem(res);
			completionItem.range = new vscode.Range(position, position);
			resolve({ items: [completionItem] });
		}

	}, 1000); // 500ms debounce delay


		// Register an inline suggestion provider
    const inlineProvider = {
        provideInlineCompletionItems: (document, position, context, token) => {
            // Return a promise that will resolve once the debounced function completes
            return new Promise(resolve => 
                debouncedFetchSuggestions(document, position, token, resolve)
  					);
        }
    };

		const providerDisposable = vscode.languages.registerInlineCompletionItemProvider(
			{ pattern: '**' }, // Apply to all files
			inlineProvider
	);

	let qaCommandDisposable = vscode.commands.registerCommand('mini-copilot.openQA', () => {
		conversationHistory = [];

		if (currentPanel) {
			currentPanel.reveal();
			return;
		}

		currentPanel = vscode.window.createWebviewPanel(
			'qaPanel',
			'Code Q&A',
			vscode.ViewColumn.Beside,
			{
				enableScripts: true,
				retainContextWhenHidden: true
			}
		);

		currentPanel.webview.html = getWebviewContent();

		currentPanel.webview.onDidReceiveMessage(
			async message => {
				console.log("Received message: ", message.command);
				switch (message.command) {
					case 'askQuestion':
						try {
							const response = await fetchChatResponse(message.text);
							console.log("Got response: ", response)
							currentPanel.webview.postMessage({ 
								command: 'response', 
								text: response 
							});
						} catch (error) {
							console.error('Error in fetchChatResponse:', error);
							currentPanel.webview.postMessage({ 
								command: 'response', 
								text: "Sorry, I encountered an error processing your request." 
							});
						}
						break;
					case 'resetConversation':
						conversationHistory = [];
						currentPanel.webview.postMessage({ 
							command: 'resetConversation' 
						});
						break;
				}
			},
			undefined,
			context.subscriptions
		);

		currentPanel.onDidDispose(
			() => {
				currentPanel = undefined;
			},
			null,
			context.subscriptions
		);
	});

	// Cmd/Ctrl+Shift+A on code
	let askWithSelectionDisposable = vscode.commands.registerCommand('mini-copilot.askWithSelection', async () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			return;
		}
	
		const selection = editor.selection;
		const selectedText = editor.document.getText(selection);
		
		if (!selectedText) {
			return;
		}
	
		// Create Q&A panel if it doesn't exist
		if (!currentPanel) {
			currentPanel = vscode.window.createWebviewPanel(
				'qaPanel',
				'Code Q&A',
				vscode.ViewColumn.Beside,
				{
					enableScripts: true,
					retainContextWhenHidden: true
				}
			);
	
			currentPanel.webview.html = getWebviewContent();
	
			currentPanel.webview.onDidReceiveMessage(
				async message => {
					switch (message.command) {
						case 'askQuestion':
							const response = await fetchChatResponse(message.text);
							currentPanel.webview.postMessage({ 
								command: 'response', 
								text: response 
							});
							break;
						case 'resetConversation':
							conversationHistory = [];
							break;
					}
				},
				undefined,
				context.subscriptions
			);
	
			currentPanel.onDidDispose(
				() => {
					currentPanel = undefined;
				},
				null,
				context.subscriptions
			);
		}
	
		// Show the panel
		currentPanel.reveal(vscode.ViewColumn.Beside);
	
		// Send the selected text to the webview
		currentPanel.webview.postMessage({ 
			command: 'insertQuestion', 
			text: `Explain this code:\n\`\`\`\n${selectedText}\n\`\`\`` 
		});
	});

	const apiKey = getOpenAIKey();
	if (!apiKey) {
		vscode.window.showWarningMessage('OpenAI API key not found. Please add it to .env file or configure in settings.');
	}

	context.subscriptions.push(disposable, providerDisposable, qaCommandDisposable, askWithSelectionDisposable);
}

function getWebviewContent() {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body {
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                color: var(--vscode-editor-foreground);
            }

            #question-input {
                width: 100%;
                min-height: 100px;
                padding: 12px;
                margin-bottom: 16px;
                border: 1px solid var(--vscode-input-border);
                background-color: var(--vscode-input-background);
                color: var(--vscode-input-foreground);
                border-radius: 6px;
                resize: vertical;
                font-family: inherit;
                font-size: 14px;
            }

            button {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
                transition: background-color 0.2s;
            }

            button:hover {
                opacity: 0.9;
            }

            #ask-button {
                background-color: var(--vscode-button-background);
                color: var(--vscode-button-foreground);
            }

            #reset-button {
                background-color: var(--vscode-errorForeground);
                color: white;
                margin-left: 10px;
            }

            #conversation {
                margin-top: 24px;
                max-height: calc(100vh - 250px);
                overflow-y: auto;
                padding-right: 12px;
            }

            .message {
                margin-bottom: 20px;
                padding: 12px 16px;
                border-radius: 8px;
                white-space: normal;
            }

            .user-message {
                background-color: var(--vscode-editor-inactiveSelectionBackground);
                margin-left: 20px;
				opacity: 0.8;
            }

            .assistant-message {
                background-color: var(--vscode-editor-selectionBackground);
                margin-right: 20px;
				opacity: 0.7;
            }

            .assistant-message pre {
                background-color: var(--vscode-editor-background);
                padding: 12px;
                border-radius: 4px;
                overflow-x: auto;
                margin: 8px 0;
				opacity: 1;
            }

            .assistant-message code {
                font-family: 'Courier New', Courier, monospace;
                font-size: 13px;
				background-color: var(--vscode-editor-background);
				padding: 2px 4px;
				border-radius: 3px;
            }

            #loading {
                display: none;
                margin: 16px 0;
                color: var(--vscode-descriptionForeground);
                font-style: italic;
                animation: pulse 1.5s infinite;
            }

			#conversation {
			    margin-top: 24px;
				max-height: calc(100vh - 250px);
				overflow-y: auto;
				padding-right: 12px;
				padding-bottom: 20px; 
			}

            @keyframes pulse {
                0% { opacity: 0.6; }
                50% { opacity: 1; }
                100% { opacity: 0.6; }
            }

            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: var(--vscode-editor-background);
            }

            ::-webkit-scrollbar-thumb {
                background: var(--vscode-scrollbarSlider-background);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--vscode-scrollbarSlider-hoverBackground);
            }
        </style>
    </head>
    <body>
        <textarea id="question-input" placeholder="Ask your coding question here... (Press Ctrl/Cmd + Enter to send)"></textarea>
        <div class="button-group">
            <button id="ask-button" onclick="askQuestion()">Ask Question</button>
            <button id="reset-button" onclick="resetConversation()">Reset Conversation</button>
        </div>
        <div id="loading">Thinking...</div>
        <div id="conversation"></div>

        <script>
            const vscode = acquireVsCodeApi();
            const conversationDiv = document.getElementById('conversation');
            const loadingDiv = document.getElementById('loading');
            const questionInput = document.getElementById('question-input');
            
            // Add keyboard shortcut
            questionInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    askQuestion();
                }
            });

            function askQuestion() {
                const question = questionInput.value;
                if (!question.trim()) return;
                
                loadingDiv.style.display = 'block';
                addToConversation('You: ' + question, 'user-message');
                questionInput.value = '';
                
                vscode.postMessage({
                    command: 'askQuestion',
                    text: question
                });
            }

            function resetConversation() {
                vscode.postMessage({ command: 'resetConversation' });
                conversationDiv.innerHTML = '';
            }

            function addToConversation(text, className) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + className;
                
                if (className === 'user-message') {
                    messageDiv.textContent = text;
                } else if (className === 'assistant-message') {
                    messageDiv.innerHTML = marked.parse(text);
                }
                
                conversationDiv.appendChild(messageDiv);
                conversationDiv.scrollTop = conversationDiv.scrollHeight;
            }

			window.addEventListener('message', event => {
				const message = event.data;
				switch (message.command) {
					case 'response':
						loadingDiv.style.display = 'none';
						addToConversation(message.text, 'assistant-message');
						break;
					case 'insertQuestion':
						questionInput.value = message.text;
						askQuestion();  // Automatically ask the question
						break;
				}
			});
        </script>
    </body>
    </html>`;
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
