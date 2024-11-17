import base64
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
path = "trained/gpt2-728"
model = AutoModelForCausalLM.from_pretrained(path)
model.to(device)
model.eval()


# Function to predict the next token
def predict_next_token(input_text, model=model, tokenizer=tokenizer, max_length=50):
    pass
    return predicted_text[input_text_len:]


def handler(event, context):
    try:
        if event.get('isBase64Encoded', False):
            body = base64.b64decode(event['body']).decode('utf-8')
        else:
            body = event['body']
    except (KeyError, json.JSONDecodeError) as e:
        return {"statusCode": 400, "body": f"Error processing request: {str(e)}"}
    return {"statusCode": 200, "body": predict_next_token(body)}

