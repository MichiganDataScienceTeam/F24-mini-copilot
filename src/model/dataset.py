from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from typing import Iterator

from preprocess import clean_comments, include, keep_only_content

from chunk import chunk

class CleanDataset(IterableDataset):
    TRAIN_SPLIT_NAME = "codeparrot/codeparrot-clean-train"
    VAL_SPLIT_NAME = "codeparrot/codeparrot-clean-valid"

    def __init__(self, train_split: bool, max_size: int = float("inf")):
        SPLIT_NAME = [CleanDataset.TRAIN_SPLIT_NAME, CleanDataset.VAL_SPLIT_NAME][int(train_split)]

        # Set max size
        self.max_size = max_size

        # Load dataset
        ds = load_dataset(SPLIT_NAME,
                          streaming=True,
                          split="train")    # Invariant for BOTH train and val sets

        # Preprocessing
        ds = ds.filter(lambda x: x["path"].endswith(".py"))               # Python only
        ds = ds.filter(lambda x: include(x["content"]))                   # DS imports only
        ds = ds.map(lambda x: {"content": clean_comments(x["content"])})  # Reformat code
        ds = ds.map(keep_only_content)                                    # Smaller samples

        # Prepare for torch DataLoader
        ds = ds.with_format("torch")

        self.ds = ds

    def generate(self) -> Iterator[dict]:
        i = iter(self.ds)
        count = 0

        while True:
            # Respect max_size
            if count == self.max_size:
                break
            count += 1

            # Yield when possible, skip and log when not
            try:
                yield next(i)
            except StopIteration:
                break
            except Exception as e:
                count -= 1
                print(f"[WARNING] Exception while loading sample {count}/{self.max_size}: {e}. Skipped item")
                continue

    def __iter__(self) -> Iterator[dict]:
        return self.generate()


class ChunkedDataset(CleanDataset):
    def __init__(self, train_split: bool, max_size: int, tokenizer: AutoTokenizer,
                 chunk_size: int = 256, chunk_overlap_len: int = 3, max_chunks: int = 128):

        super().__init__(train_split, max_size)

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlapping_len = chunk_overlap_len
        self.max_chunks = max_chunks

    def generate(self) -> Iterator[dict]:
        count = 0

        for text in super().generate():
            # Attempt to chunk each code sample
            chunks = None
            try:
                chunks = chunk(inp=text["content"],
                               tokenizer=self.tokenizer, 
                               chunk_size=self.chunk_size,
                               overlapping_len=self.overlapping_len,
                               max_chunks=self.max_chunks)
            except Exception as e:
                print(f"[WARNING] Exception while chunking sample {count}/{self.max_size}: {e}. Skipped item")
                continue

            # Extract input ids and attention masks
            ids, mask = chunks["input_ids"], chunks["attention_mask"]

            # Yield each chunk, stopping if max_size is reached
            for i in range(ids.size()[0]):
                # Stop yielding if max_size is reached
                if count >= self.max_size:
                    break

                # Yield
                yield {
                    "input_ids": ids[i],
                    "attention_mask": mask[i]
                }
                count += 1

            # Stop generating new chunks if max_size is reached
            if count >= self.max_size:
                break


# SAMPLE USAGE
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_10M")

    ds = ChunkedDataset(
        train_split=True,      # Use training split
        max_size=1_000_000,    # Provide up to 1 million samples (not files)
        tokenizer=tokenizer,   # Set tokenizer
        chunk_size=256,        # Max length of id/mask sequences is 256
        chunk_overlap_len=3,   # Chunks share 3 ids with the previous chunk
        max_chunks=128,        # Max chunks per file
    )

    # ChunkedDataset is iterable, so it can be directly passed to a DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset=ds,
        batch_size=16,
        # shuffle should NOT be set because the dataset has unknown length
    )

    # Inspect a single element of this batch
    for batch in loader:
        print(batch)
        break
=======
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