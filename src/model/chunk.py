## This file contains the function to chunk the input text into smaller pieces for the model to process
## The input text is tokenized and then split into chunks of size chunk_size

from transformers import AutoTokenizer
import torch

def chunk(inp: str, 
          tokenizer: AutoTokenizer, 
          chunk_size: int = 256, 
          overlapping_len: int = 3,
          max_chunks: int = 128) -> torch.Tensor:
    
    # Tokenize entire sample
    tokenized_txt = tokenizer(inp,
                              return_tensors="pt"
                             )["input_ids"].view(-1)
    token_len = len(tokenized_txt)
    
    # Add chunks
    chunks = []
    last_padding_size = 0

    for i in range(0, token_len-overlapping_len, chunk_size-overlapping_len):
        # Exit if max_chunks is met
        if len(chunks) >= max_chunks:
            break

        # Create (potentially too short) new chunk
        new_chunk = tokenized_txt[i:i+chunk_size]
        
        # Generate (potentially empty) padding
        padding = torch.full(
            size=(chunk_size - len(new_chunk), ),
            fill_value=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
        last_padding_size = max(0, chunk_size - len(new_chunk))

        # Pad
        new_chunk = torch.cat((new_chunk, padding))
        
        # Add new correctly-sized chunk
        chunks.append(new_chunk)
    
    # Compile results
    input_ids = torch.stack(chunks)
    attention_mask = torch.ones_like(input_ids)

    if last_padding_size >= 2:
        attention_mask[-1, -last_padding_size:] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
