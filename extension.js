// Import necessary modules
const vscode = require('vscode');

function activate(context) {

    const disposable = vscode.commands.registerCommand('mini-copilot.helloWorld', function () {
        vscode.window.showInformationMessage('Hello World from Inline Completion!');
    });

    console.log('Congratulations, your extension "inline-completion" is now active!');

    // Debounce function for async functions
    function debounceAsync(func, wait) {
        let timeout = null;
        let lastArgs = null;
        let pendingPromise = null;
        let pendingResolve = null;
        let pendingReject = null;

        return function (...args) {
            // Always capture the latest arguments
            lastArgs = args;

            // Clear the existing timeout
            if (timeout) clearTimeout(timeout);

            // Create a single pending promise if it doesn't exist
            if (!pendingPromise) {
                pendingPromise = new Promise((resolve, reject) => {
                    pendingResolve = resolve;
                    pendingReject = reject;
                });
            }

            // Set up the new timeout
            timeout = setTimeout(async () => {
                try {
                    // Execute the function with the latest arguments
                    const result = await func(...lastArgs);
                    pendingResolve(result);
                } catch (error) {
                    pendingReject(error);
                } finally {
                    // Reset everything
                    timeout = null;
                    pendingPromise = null;
                    pendingResolve = null;
                    pendingReject = null;
                }
            }, wait);

            // Return the pending promise
            return pendingPromise;
        };
    }


    // Function to fetch data from your API
    async function getData(inputText) {
        const url = "https://fjru3p67wsonsdfbdp7p673giq0cvtep.lambda-url.us-east-2.on.aws/";

        try {
            const response = await fetch(url, {
                method: "POST",
                body: {"body": inputText},
                headers: {
                    'Content-Type': 'text/plain',
                },
            });

            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`);
            }

            const data = await response.text();
            return data;
        } catch (error) {
            console.error(error.message);
            throw error;
        }
    }

    // Inline Completion Provider
    const inlineCompletionProvider = {
        provideInlineCompletionItems: debounceAsync(async (
            document,
            position,
            _context,
            _token
        ) => {
            // Get the text before the cursor
            const startPosition = position.with(undefined, Math.max(0, position.character - 5));
            const textBeforeCursor = document.getText(new vscode.Range(startPosition, position));

            try {
                // Fetch the completion from your API
                const completionText = await getData(textBeforeCursor);

                // Create an InlineCompletionItem
                const completionItem = new vscode.InlineCompletionItem(completionText);

                // Return the completion item
                return [completionItem];
            } catch (error) {
                console.error('Error fetching completion:', error);
                return null;
            }
        }, 1000) // Debounce with 1 second delay
    };


    // Register the provider
    const providerDisposable = vscode.languages.registerInlineCompletionItemProvider(
        { pattern: '**' },
        inlineCompletionProvider
    );

    context.subscriptions.push(disposable, providerDisposable);
}

exports.activate = activate;

function deactivate() {
    // Clean up resources if necessary
}

exports.deactivate = deactivate;
