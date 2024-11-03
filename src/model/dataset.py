import os
import torch
import evaluate
import regex as re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, pipeline
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, IterableDataset
# these are all the libraries you'd need


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'Using device: {device}')

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))  # Should return the name of the GPU

    import re

def clean_data(inp: str) -> str:
    """OPTIONAL: Perform data cleaning, if necessary."""
    s = re.sub(r'^#.*\n?', '', inp, flags=re.MULTILINE)
    return s

def get_data() -> Dataset:
    # https://huggingface.co/datasets/codeparrot/codeparrot-clean
    # Load the dataset
    ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, trust_remote_code=True, split="train")

    # Clean the data
    ds = ds.map(lambda x: {"content": clean_data(x["content"])})

    return ds



def get_train_valid_data(dataset: Dataset) -> (Dataset, Dataset):
    """TODO: Split the dataset into training and validation sets."""
    # This is not too straightforward because the dataset is a streaming dataset
    #n = 300000
    n = 150
    split = int(n*0.75)
    dataset.shuffle()
    ds_train = dataset.take(n)
    ds_valid = ds_train.skip(split)
    ds_train = ds_train.take(split)
    return ds_train, ds_valid


class SafeIterableDataset(torch.utils.data.IterableDataset):
    """Wrapper to account for download errors so training doesn't stop due to error pulling data from HF."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                item = next(iterator)
                yield item
            except StopIteration:
                break
            except Exception as e:
                print(f"Caught exception during data loading: {e}. Skipping item.")
                continue

def tokenize(inp: list[str]):
    """
    TODO: Tokenize the input.
    Consider:
    - Padding?
    - Truncation?
    - Anything else?
    """
    # truncate to first 256 tokens
    # pad to make every example the same size (ex: 256 tokens)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inp = tokenizer(inp)['input_ids']
    results = []

    for ex in inp:
        ex.extend([0] * (max(0, 256 - len(ex))))
        ex = ex[:256]
        results.append(torch.tensor(ex))
    return torch.stack(results)


    #return(tokenizer(inp)["input_ids"])

# TODO: Consider setting up model checkpointing (set up a directory to save checkpoints)

def get_dataloaders(batch_size = 16):
    batch_size = 16
    os.environ["HF_HUB_ETAG_TIMEOUT"]     = "500"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "500"
    model     = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay= 0.001)

    model.to(device)

    dataset = get_data()
    train_data, valid_data = get_train_valid_data(dataset)
    train_data = SafeIterableDataset(train_data)
    valid_data = SafeIterableDataset(valid_data)

    train_loader = DataLoader(train_data,  batch_size=batch_size)
    valid_loader  = DataLoader(valid_data,  batch_size=batch_size)
    return train_loader, valid_loader
