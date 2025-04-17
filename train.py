# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Optional: for progress bar
import config
from model import VGG16Model  # Import your model class (use the actual name)
# Import your dataloaders (use the actual variable names)
from dataset import your_train_loader, your_valid_loader


def my_descriptively_named_train_function():

    print(f"Using device:{config.DEVICE}")
    device = torch.device(config.DEVICE)

    model = VGG16Model(num_classes=config.NUM_CLASSES).to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.OPTIMIZER_MOMENTUM,
        weight_decay=config.OPTIMIZER_WEIGHT_DECAY
    )

    print("Starting Training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        # Use tqdm for progress bar (optional)
        train_pbar = tqdm(
            your_train_loader,
            desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")

        for inputs, label in train_pbar:
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()

            # fwd pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.items() * inputs.size(0)

            train_pbar.set_postfix({'loss': loss.item()})

        # Use sampler length if using SubsetRandomSampler
        epoch_loss = running_loss / len(your_train_loader.sampler)
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Training Loss: {epoch_loss:.4f}")

        # --- Optional: Validation Step ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        valid_pbar = tqdm(
            your_valid_loader,
            desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in valid_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                valid_pbar.set_postfix({'val_loss': loss.item()})

        epoch_val_loss = val_loss / len(your_valid_loader.sampler)
        epoch_val_acc = 100 * correct / total
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_acc:.2f}%")
        # --- End Optional Validation Step ---

        print("Finished Training")

        # Save the trained model weights
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)
        print(f"Model weights saved to {config.CHECKPOINT_PATH}")

    return model  # Optional: return the trained model


if __name__ == '__main__':
    my_descriptively_named_train_function()
