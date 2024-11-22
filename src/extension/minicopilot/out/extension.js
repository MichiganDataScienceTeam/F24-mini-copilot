"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
// Debounce utility function
function debounce(func, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), delay);
    };
}
// This method is called when your extension is activated
function activate(context) {
    console.log('Congratulations, your extension "minicopilot" is now active!');
    const disposable = vscode.commands.registerCommand('minicopilot.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from minicopilot!');
    });
    // Debounced function to fetch suggestions
    const debouncedFetchSuggestions = debounce(async (document, position, token, resolve) => {
        const text = document.getText(new vscode.Range(new vscode.Position(Math.max(0, position.line - 5), 0), position));
        const route = "https://fjru3p67wsonsdfbdp7p673giq0cvtep.lambda-url.us-east-2.on.aws";
        // const route = "http://127.0.0.1:5000/completion";
        console.log("in debounce");
        if (text.trim() !== '') {
            const body = text;
            try {
                const response = await fetch(route, {
                    method: 'POST',
                    body: body,
                    headers: { 'Content-Type': 'text/plain' },
                });
                if (response.ok) {
                    const responseBody = await response.text();
                    const completionItem = new vscode.InlineCompletionItem(responseBody);
                    completionItem.range = new vscode.Range(position, position);
                    resolve({ items: [completionItem] });
                }
                else {
                    console.error('Error:', response.status);
                    resolve({ items: [] });
                }
            }
            catch (error) {
                console.error('Fetch error:', error);
                resolve({ items: [] });
            }
        }
        else {
            resolve({ items: [] });
        }
    }, 1000); // 500ms debounce delay
    // Register an inline suggestion provider
    const inlineProvider = {
        provideInlineCompletionItems: (document, position, context, token) => {
            // Return a promise that will resolve once the debounced function completes
            return new Promise(resolve => {
                debouncedFetchSuggestions(document, position, token, resolve);
            });
        }
    };
    const providerDisposable = vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, // Apply to all files
    inlineProvider);
    context.subscriptions.push(disposable, providerDisposable);
}
// This method is called when your extension is deactivated
function deactivate() { }
//# sourceMappingURL=extension.js.map