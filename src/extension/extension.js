// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');

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
	const disposable = vscode.commands.registerCommand('mini-copilot.helloWorld', function () {
		console.log("activated!");
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

	context.subscriptions.push(disposable, providerDisposable);
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
