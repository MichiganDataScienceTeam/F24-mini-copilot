# Mini Copilot, Fall 2024

In this project, we will attempt to recreate GitHub Copilot. Since properly recreating the service requires an immense amount of compute for model training, we will seek to build a **Mini Copilot**.

This repository contains both educational materials (starter notebooks for basic training) and our actual implementation of recreating GitHub Copilot. Education materials can be found under `education/`, and our implementation can be found under `src/`.

Accompanying educational slide decks can be found [here](https://drive.google.com/drive/folders/1EUTBDfIL_Y3dRKkDsNaysntJQjiOef1c?usp=drive_link) (requires UMich login).

## Methods

Recreating GitHub Copilot involves various parts -- the underlying code completion model, the endpoint serving the model, and the actual application (VSCode extension) using the model.

- **Code completion model:** To train the underlying code completion model, we train `gpt-2` on a code completion task (code found under `src/model`).
- **Serving the model:** To serve the model, we use AWS Lambda (container image found under `src/backend`).
- **VSCode extension:** To build the VSCode extension, we use the `registerInlineCompletionItemProvider` VSCode API (full implementation found under `src/extension`). This extension calls our Lambda endpoint to supply the code completion.

## Project Master Schedule

| Week | Date   | Topic                                      |
|------|--------|--------------------------------------------|
| 1    | 9/22   | Project Overview + Causal Language Modeling (CLM) w/ n-grams |
| 2    | 9/29   | CLM + High Performance Computing (HPC)      |
| 3    | 10/6   | CLM Continued + Model Evaluation            |
| 4    | 10/20  | Masked Language Modeling (MLM)              |
| 5    | 10/27  | Model Deployment                            |
| 6    | 11/3   | Creating a VSCode Extension                 |
| 7    | 11/10  | Buffer Week / Going deeper                  |
| 8    | 11/17  | Final Expo Prep                             |

## Acknowledgements

Project Members:
- Aarushi Shah
- Colin Gordon
- Dennis Farmer
- Jeffrey Lu
- Kevin Han
- Maxim Kim
- Michael Liao
- Rafael Khoury
- Selina Sun

Project Leads:
- Amirali Danai
- Nishant Dash
