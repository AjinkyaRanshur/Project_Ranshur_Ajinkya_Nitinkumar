import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MyModel            # Your model definition
from dataset import YourDataset      # Your dataset class
from config import Config            # Config file for hyperparameters


def train_model(checkpoint_path: str = "checkpoint.pth") -> None:
    """
    Trains the model from scratch and saves weights to checkpoint_path.
    """
    # Load config
    config = Config()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset and dataloader
    train_dataset = YourDataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # Initialize model, loss, optimizer
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {epoch_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model trained and saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model and save a checkpoint.")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if checkpoint already exists."
    )
    args = parser.parse_args()

    checkpoint_path = "checkpoint.pth"
    # If checkpoint exists and user did not force retrain, skip training
    if os.path.exists(checkpoint_path) and not args.force_retrain:
        print(f"Checkpoint '{checkpoint_path}' exists. Skipping training.")
    else:
        train_model(checkpoint_path)
    print("Training step complete. You can now run interface.py for evaluation.")

