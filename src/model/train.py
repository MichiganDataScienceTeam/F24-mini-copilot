import os
import torch
import evaluate
import regex as re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, pipeline
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, IterableDataset
from dataset import get_dataloaders, tokenize
# import tokenize from tokenizer.py


def make_model(lr, weight_decay, optim=torch.optim.Adam):
    """Make a model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    optimizer = optim(params=model.parameters(), lr=1e-3, weight_decay= 0.001)

    model.to(device)
    return model, optimizer


# train one epoch
def train_single_epoch(model, tokenize, optimizer, train_loader):
    """Take in model, test_loader"""
    model.train()
    for batch in train_loader:
        # Note that device that data is on should be the same as the model
        input_ids = tokenize(batch["content"])
        labels = input_ids.clone()
        # labels are automatically shifted for next token prediction
        # assuming model is of type AutoModelForCausalLM
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        #raise NotImplementedError


def validate(model, tokenize, test_loader):
    """
    Run validation across batches in test_loader
    'Take in model, test_loader and batch'

    """
    losses=[]
    model.eval()

    with torch.no_grad():
        # for batch in test_loader:
        #     # TODO: Implement validation loop
        #     # Note that device that data is on should be the same as the model
        #     ...
        for batch in test_loader:
            # Implement validation loop
            # Note that device that data is on should be the same as the model
            input_ids = tokenize(batch["content"])
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            losses.append(outputs.loss)
            loss = torch.mean(torch.tensor(losses))
            try:
                perplexity = torch.exp(loss)
            except OverflowError:
                perplexity = float("inf")
                raise NotImplementedError
            return loss.item(), perplexity.item()
        
            # raise NotImplementedError


# TODO: Consider setting up model checkpointing (set up a directory to save checkpoints)

# train for many epochs
def train(n_epochs, model, tokenizer, optimizer, train_loader, save_interval=1, save_dir='checkpoints'):
    """
    train model for n_epochs
    """
    model.train()
    
    os.makedirs(save_dir, exist_ok=True)
    # Clear residual gradients (might cause issues with taking grad. of frozen layers)
    model.zero_grad(set_to_none=True)

    for epoch in range(n_epochs):
        print(f"Epoch: {epoch}")
        train_single_epoch(model, tokenizer, optimizer, train_loader)
        if (epoch + 1) % save_interval == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
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
        return -1
    
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model     = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay= 0.001)
    train_dataloader, valid_dataloader = get_dataloaders(batch_size=16)
    train(
        n_epochs=5,
        model=model,
        tokenizer=tokenize,
        optimizer=optimizer,
        train_loader=train_dataloader
    )


if __name__ == "__main__":
    main()