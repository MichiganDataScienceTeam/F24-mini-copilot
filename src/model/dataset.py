from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from preprocess import clean_comments, include

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

        # Prepare for torch DataLoader
        ds = ds.with_format("torch")

        self.ds = ds
    
    def generate(self) -> dict:
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

    def __iter__(self):
        return iter(self.generate())
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)

