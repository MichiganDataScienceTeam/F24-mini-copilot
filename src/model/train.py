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
def train(n_epochs, model, tokenizer, optimizer, train_loader):
    """
    train model for n_epochs
    """
    model.train()
    
    # Clear residual gradients (might cause issues with taking grad. of frozen layers)
    model.zero_grad(set_to_none=True)

    for epoch in range(n_epochs):
        print(f"Epoch: {epoch}")
        train_single_epoch(model, tokenizer, optimizer, train_loader)

    print("Training complete")


# TODO: Save the model
