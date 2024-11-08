import os
import math
import argparse

from tqdm import tqdm
import evaluate

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dataset import ChunkedDataset


# TODO: Maybe consider gradient accumulation if GPU memory restricts batch sizes too much
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


# TODO: Consider setting up model checkpointing (set up a directory to save checkpoints)

# train for many epochs
def train(model: AutoModelForCausalLM,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          n_epochs: int,
          save_interval: int,
          checkpoint_dir: str = 'checkpoints',
          custom_checkpoint: str=None):
    model.train()

    if custom_checkpoint:
        model = torch.load(custom_checkpoint)
    
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

    print("Training complete")

def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    """
    Load the latest checkpoint from the specified directory.
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        
    Returns:
        int: The epoch number of the loaded checkpoint, or -1 if no checkpoint was found.
    """
    # Get a list of all .pt files in the checkpoint directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        return 0

    
    # Determine the most recent checkpoint file based on modification time
    latest_checkpoint = max(
        (os.path.join(checkpoint_dir, f) for f in checkpoint_files),
        key=os.path.getmtime
    )
    
    print(f"Loading checkpoint from: {latest_checkpoint}")
    
    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint)
    
    # Load the model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the epoch number for further reference
    return checkpoint['epoch']

def main(n_epochs: int,
         save_interval: int,
         checkpoint_dir: str,
         custom_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    dataset_config = {
        "max_size": 10_000_000,  # Set arbitrarily, TODO: pick a number more intentionally
        "tokenizer": tokenizer,
        "chunk_size": 1024,      # IIRC max input length, TODO: verify
        "chunk_overlap_len": 3,  # Set arbitrarily, TODO: pick a number more intentionally
        "max_chunks": 512        # Set arbitrarily, TODO: pick a number more intentionally
    }

    batch_size=16 # Set arbitrarily, TODO: pick a number more intentionally

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-3, 
        weight_decay=0.001)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description="Train Copilot Model")
    parser.add_argument("--checkpoint-folder", default="checkpoints", help="Directory where checkpoints are to be stored")
    parser.add_argument("-c", "--custom-checkpoint", default=None, help="Checkpoint to load from (checkpoint.pt)")
    parser.add_argument("-n", "--epochs", default=5, help="Number of epochs to train for")
    parser.add_argument("-i", "--save-interval", default=5, help="Epoch interval between checkpoint saves")
    args = parser.parse_args()
    
    main(int(args.epochs), int(args.save_interval), args.checkpoint_folder, args.custom_checkpoint)