import * as vscode from 'vscode';

// Debounce utility function
function debounce(func: (...args: any[]) => void, delay: number) {
    let timeout: NodeJS.Timeout;
    return (...args: any[]) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), delay);
    };
}

// This method is called when your extension is activated
export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "minicopilot" is now active!');

    const disposable = vscode.commands.registerCommand('minicopilot.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from minicopilot!');
    });

    // Debounced function to fetch suggestions
    const debouncedFetchSuggestions = debounce(async (document, position, token, resolve) => {
        const text = document.getText(new vscode.Range(new vscode.Position(Math.max(0, position.line - 5), 0), position));
        const route = "https://fjru3p67wsonsdfbdp7p673giq0cvtep.lambda-url.us-east-2.on.aws";
        // const route = "http://127.0.0.1:5000/completion";
        console.log("in debounce")

        if (text.trim() !== '') {
            const body = text;

            try {
                const response = await fetch(route, {
                    method: 'POST',
                    body: body,
                    headers: { 'Content-Type': 'text/plain' },
                });

                if (response.ok) {
                    const responseBody: any = await response.text();
                    const completionItem = new vscode.InlineCompletionItem(responseBody);
                    completionItem.range = new vscode.Range(position, position);
                    resolve({ items: [completionItem] });
                } else {
                    console.error('Error:', response.status);
                    resolve({ items: [] });
                }
            } catch (error) {
                console.error('Fetch error:', error);
                resolve({ items: [] });
            }
        } else {
            resolve({ items: [] });
        }
    }, 1000); // 500ms debounce delay

    // Register an inline suggestion provider
    const inlineProvider: vscode.InlineCompletionItemProvider = {
        provideInlineCompletionItems: (document, position, context, token) => {
            // Return a promise that will resolve once the debounced function completes
            return new Promise(resolve => {
                debouncedFetchSuggestions(document, position, token, resolve);
            });
        }
    };

    const providerDisposable = vscode.languages.registerInlineCompletionItemProvider(
        { pattern: '**' }, // Apply to all files
        inlineProvider
    );

    context.subscriptions.push(disposable, providerDisposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}
