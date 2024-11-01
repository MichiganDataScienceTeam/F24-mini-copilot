"""
Do not put this version in a pull request without review
Translated from a notebook
"""

from transformers import AutoTokenizer
from dataset import CleanDataset

MAX_SIZE = 100
VOCAB_SIZE = 52000
OUT_FILE_PATH = "./retrained_attempt_1"

# Get dataset
ds = CleanDataset(
    train_split=True,
    max_size=MAX_SIZE
)

# Reformat data for retraining
def get_training_corpus(ds_):
    for sample in ds_:
        yield sample["content"]

# Load old tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer = old_tokenizer.train_new_from_iterator(
    text_iterator=get_training_corpus(ds),
    vocab_size=VOCAB_SIZE
)

# Save tokenizer
tokenizer.save_pretrained(OUT_FILE_PATH)
