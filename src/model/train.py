import os
import math
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import ChunkedDataset


def train_single_epoch(model: AutoModelForCausalLM,
                       optimizer: torch.optim.Optimizer,
                       train_loader: DataLoader):
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        
        loss = outputs.loss
        loss.backward()

        optimizer.step()


def validate(model: AutoModelForCausalLM,
             test_loader: DataLoader) -> tuple[float, float]:
    loss_sum = 0
    n_losses = 0

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )

            loss_sum += outputs.loss.item()
            n_losses += 1
        
    loss = loss_sum/n_losses

    perplexity = math.exp(loss) # Default returns inf when overflow

    return loss, perplexity


def train(model: AutoModelForCausalLM,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          n_epochs: int,
          save_interval: int,
          checkpoint_dir: str,
          custom_checkpoint: str):
    if custom_checkpoint:
        model = torch.load(custom_checkpoint)
    
    model.train()
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, n_epochs+1):
        print(f"Epoch: {epoch}")
        
        train_single_epoch(model, optimizer, train_loader)

        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            print(f"Checkpoint saved at {checkpoint_path}")
        
        # TODO: print training & validation metrics

    print("Training complete")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset_config = {
        "max_size": 10_000_000,  # Set arbitrarily, TODO: pick a number more intentionally
        "tokenizer": tokenizer,
        "chunk_size": 1024,      # IIRC max input length, TODO: verify
        "chunk_overlap_len": 3,  # Set arbitrarily, TODO: pick a number more intentionally
        "max_chunks": 512        # Set arbitrarily, TODO: pick a number more intentionally
    }

    # TODO: Consider gradient accumulation if GPU memory restricts batch sizes too much
    batch_size=16 # Set arbitrarily, TODO: pick a number more intentionally

    train_loader = DataLoader(ChunkedDataset(
        train_split=True,
        **dataset_config
    ), batch_size=batch_size)

    valid_loader = DataLoader(ChunkedDataset(
        train_split=False,
        **dataset_config
    ), batch_size=batch_size)

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        n_epochs=n_epochs,
        save_interval=save_interval,
        checkpoint_dir=checkpoint_dir,
        custom_checkpoint=custom_checkpoint
    )

    # TODO: validation


if __name__ == "__main__":
    main()
