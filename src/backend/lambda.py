import base64
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("trained/gpt2-728")
model.to(device)
model.eval()

pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer, device=device, max_new_tokens=32)


# Function to predict the next token
def get_completion(inp: str) -> str:
    return pipe(inp)[0]["generated_text"][len(inp):]


# Lambda handler
def handler(event, context):
    try:
        if event.get('isBase64Encoded', False):
            body = base64.b64decode(event['body']).decode('utf-8')
        else:
            body = event['body']
    except (KeyError, json.JSONDecodeError) as e:
        return {"statusCode": 400, "body": f"Error processing request: {str(e)}"}
    return {"statusCode": 200, "body": get_completion(body)}
