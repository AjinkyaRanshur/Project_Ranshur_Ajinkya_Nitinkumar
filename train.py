#train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import VGG16Model
from dataset import train_loader, test_loader
import config
from tqdm import tqdm  # For progress bars

def save_training_plot(train_losses, val_accuracies, filename="training_plot.png"):
    """
    Create and save a plot showing training loss and validation accuracy.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    
    # Plot validation accuracy
    ax2.plot(val_accuracies, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Training plot saved as {filename}")

def my_descriptively_named_train_function(checkpoint_path=config.CHECKPOINT_PATH):
    """
    Trains the VGG16 model on CIFAR100 dataset and saves the best model.
    Returns the trained model.
    """
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = VGG16Model(num_classes=config.NUM_CLASSES).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Switch to Adam optimizer which often works better for deep networks
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.OPTIMIZER_WEIGHT_DECAY
    )
    
    # Learning rate scheduler - use CosineAnnealingLR which works well with Adam
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )
    
    # Training loop
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Create a progress bar for the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", 
                    ncols=100, unit="batch")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        # Create a progress bar for validation
        val_pbar = tqdm(test_loader, desc="Validating", 
                        ncols=100, unit="batch", leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = 100 * correct / total
                val_pbar.set_postfix({"accuracy": f"{current_acc:.2f}%"})
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Update learning rate based on scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.2f}% LR: {current_lr:.6f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Improved accuracy: {best_accuracy:.2f}% - Saved model to {checkpoint_path}")
    
    # Save training plot
    save_training_plot(train_losses, val_accuracies)
    
    # Load best model for return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the VGG16 model on CIFAR100 dataset.")
    parser.add_argument(
        "--force-retrain", 
        action="store_true",
        help="Retrain even if checkpoint already exists."
    )
    args = parser.parse_args()
    
    # Check if checkpoint exists and retrain if needed
    if os.path.exists(config.CHECKPOINT_PATH) and not args.force_retrain:
        print(f"Checkpoint '{config.CHECKPOINT_PATH}' exists. Skipping training.")
        print("Use --force-retrain to retrain the model.")
    else:
        my_descriptively_named_train_function(config.CHECKPOINT_PATH)
    
    print("Training step complete. You can now run interface.py for evaluation.")
