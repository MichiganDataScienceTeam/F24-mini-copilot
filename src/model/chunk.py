## This file contains the function to chunk the input text into smaller pieces for the model to process
## The input text is tokenized and then split into chunks of size chunk_size

from transformers import AutoTokenizer
import torch

def chunk(inp: str, 
          tokenizer: AutoTokenizer, 
          chunk_size: int = 256, 
          overlapping_len: int = 3,
          num_chunks: int = 128) -> list:
    tokenized_txt = tokenizer(inp, return_tensors="pt")["input_ids"]
    token_len = tokenized_txt.size()[1]
    
    if token_len <= chunk_size:
        return [tokenized_txt]
    else:
        chunks = []
        chunks.append(tokenized_txt.view(-1)[:chunk_size])
        for i in range(chunk_size, token_len, chunk_size):
            if len(chunks) == num_chunks:
                # only return num_chunks chunks so that it is not stuck on the same sample for too long
                break
            if i + chunk_size > token_len:
                new_chunk = tokenized_txt.view(-1)[i-overlapping_len:]
                new_chunk_len = new_chunk.size()[0]
                new_chunk = new_chunk.view(1, new_chunk_len)
                pad_len = chunk_size - new_chunk_len
                padded_chunk = torch.cat((new_chunk, torch.full((1, pad_len), tokenizer.eos_token_id)), dim=1)
                padded_chunk = padded_chunk.view(-1)
                chunks.append(padded_chunk)
                continue
            else:
                new_chunk = tokenized_txt.view(-1)[i-overlapping_len:i+chunk_size-overlapping_len]
                chunks.append(new_chunk)
        return chunks