// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode')
const openai = require('openai')
const path = require('path')
const fs = require("fs")
require('dotenv').config({ path: path.join(__dirname, '.env') })

console.log('API Key exists:', !!process.env.OPENAI_API_KEY)

let currentPanel
let conversationHistory = []

// Add OpenAI client configuration
const openaiClient = new openai.OpenAI({
    apiKey: process.env.OPENAI_API_KEY || getOpenAIKey() 
})

// Debounce utility function
function debounce(func, delay) {
	let timeout
	return (...args) => {
		clearTimeout(timeout)
		timeout = setTimeout(() => func(...args), delay)
	}
}

async function fetchPrediction(text) {
	console.log(`Prediction requested at ${new Date()}`)

	if (text.trim() == "") {
		console.log("Request aborted: Empty text")
		console.log("Response:")
		return ""
	}

	const route = "https://fjru3p67wsonsdfbdp7p673giq0cvtep.lambda-url.us-east-2.on.aws"
	const body = text.trim()

	try {
		const response = await fetch(route, {
			method: 'POST',
			body: body,
			headers: { 'Content-Type': 'text/plain' },
		})
		
		if (response.ok) {
			const res = await response.text()
			
			console.log(`Successfully requested: ${body}`)
			console.log(`Response: ${res}`)
			return res
		} else {
			console.log("Request failed: Encountered error while fetching")
			console.log("Response:")
			console.error(`Error: ${response.status}`)
			return ""
		}
	} catch (error) {
		console.log("Request failed: Encountered error while fetching")
		console.log("Response:")
		console.error(`Fetch Error: ${error}`)
		return ""
	}
}

async function fetchChatResponse(question) {
    try {
        if (conversationHistory.length === 0) {
            conversationHistory.push({
                role: "system",
                content: "You are a helpful coding assistant. When answering follow-up questions, maintain context from the previous conversation and understand that 'What about X' usually means to apply the previous concept to X."
            })
        }

		conversationHistory.push({ role: "user", content: question })
		console.log("Current conversation:", conversationHistory)
		
        const completion = await openaiClient.chat.completions.create({
            model: "gpt-4o-mini",
            messages: conversationHistory,
            temperature: 0.9,
			max_tokens: 500,
			presence_penalty: 0.6,
			frequency_penalty: 0.6
        })

        const response =  completion.choices[0].message.content
		conversationHistory.push({ role: "assistant", content: response })

		console.log("OpenAI response received")
		return response
    } catch (error) {
        console.error('OpenAI API error:', error)
        return "Sorry, I encountered an error processing your request."
    }
}

function getOpenAIKey() {
	if (process.env.OPENAI_API_KEY) return process.env.OPENAI_API_KEY

    const config = vscode.workspace.getConfiguration('miniCopilot')
    return config.get('openAIKey')
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */

function activate(context) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "tutorial" is now active!')

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with  registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('mini-copilot.helloWorld', function () {
		console.log("activated!")
		vscode.window.showInformationMessage('Hello World from mini-copilot!')
	})

	const debouncedFetchSuggestions = debounce(async (document, position, token, resolve) => {
		console.log("triggered")
		const text = document.getText(new vscode.Range(new vscode.Position(Math.max(0, position.line - 5), 0), position))
		const res = await fetchPrediction(text)
		if (res.length == 0) {
			resolve({ items: [] })
		} else {
			const completionItem = new vscode.InlineCompletionItem(res)
			completionItem.range = new vscode.Range(position, position)
			resolve({ items: [completionItem] })
		}

	}, 1000) // 500ms debounce delay


		// Register an inline suggestion provider
    const inlineProvider = {
        provideInlineCompletionItems: (document, position, context, token) => {
            // Return a promise that will resolve once the debounced function completes
            return new Promise(resolve => 
                debouncedFetchSuggestions(document, position, token, resolve)
  					)
        }
    }

		const providerDisposable = vscode.languages.registerInlineCompletionItemProvider(
			{ pattern: '**' }, // Apply to all files
			inlineProvider
	)

	let qaCommandDisposable = vscode.commands.registerCommand('mini-copilot.openQA', () => {
		conversationHistory = []

		if (currentPanel) {
			currentPanel.reveal()
			return
		}

		currentPanel = vscode.window.createWebviewPanel(
			'qaPanel',
			'Code Q&A',
			vscode.ViewColumn.Beside,
			{
				enableScripts: true,
				retainContextWhenHidden: true
			}
		)

		currentPanel.webview.html = getWebviewContent()

		currentPanel.webview.onDidReceiveMessage(
			async message => {
				console.log("Received message: ", message.command)
				switch (message.command) {
					case 'askQuestion':
						try {
							const response = await fetchChatResponse(message.text)
							console.log("Got response: ", response)
							currentPanel.webview.postMessage({ 
								command: 'response', 
								text: response 
							})
						} catch (error) {
							console.error('Error in fetchChatResponse:', error)
							currentPanel.webview.postMessage({ 
								command: 'response', 
								text: "Sorry, I encountered an error processing your request." 
							})
						}
						break
					case 'resetConversation':
						conversationHistory = []
						currentPanel.webview.postMessage({ 
							command: 'resetConversation' 
						})
						break
				}
			},
			undefined,
			context.subscriptions
		)

		currentPanel.onDidDispose(
			() => {
				currentPanel = undefined
			},
			null,
			context.subscriptions
		)
	})

	// Cmd/Ctrl+Shift+A on code
	let askWithSelectionDisposable = vscode.commands.registerCommand('mini-copilot.askWithSelection', async () => {
		const editor = vscode.window.activeTextEditor
		if (!editor) {
			return
		}
	
		const selection = editor.selection
		const selectedText = editor.document.getText(selection)
		
		if (!selectedText) {
			return
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
			)
	
			currentPanel.webview.html = getWebviewContent()
	
			currentPanel.webview.onDidReceiveMessage(
				async message => {
					switch (message.command) {
						case 'askQuestion':
							const response = await fetchChatResponse(message.text)
							currentPanel.webview.postMessage({ 
								command: 'response', 
								text: response 
							})
							break
						case 'resetConversation':
							conversationHistory = []
							break
					}
				},
				undefined,
				context.subscriptions
			)
	
			currentPanel.onDidDispose(
				() => {
					currentPanel = undefined
				},
				null,
				context.subscriptions
			)
		}
	
		// Show the panel
		currentPanel.reveal(vscode.ViewColumn.Beside)
	
		// Send the selected text to the webview
		currentPanel.webview.postMessage({ 
			command: 'insertQuestion', 
			text: `Explain this code:\n\`\`\`\n${selectedText}\n\`\`\`` 
		})
	})

	const apiKey = getOpenAIKey()
	if (!apiKey) {
		vscode.window.showWarningMessage('OpenAI API key not found. Please add it to .env file or configure in settings.')
	}

	context.subscriptions.push(disposable, providerDisposable, qaCommandDisposable, askWithSelectionDisposable)
}

function getWebviewContent() {
    return fs.readFileSync(path.join(__dirname, 'webview.html'), "utf-8")
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
