#interface.py

# --- IMPORTANT: Modify the names on the left (before 'import') ---
# --- and the names on the right (after 'import' and before 'as') ---
# --- to match YOUR actual class/function/variable names ---

# Import your Model Class
from model import VGG16Model as TheModel

# Import your Training Function
from train import my_descriptively_named_train_function as the_trainer

# Import your Prediction Function
from predict import cryptic_inf_f as the_predictor

# Import your Dataset Class
# Creating a simple wrapper class for the datasets
class YourCifar100Dataset:
    """Wrapper class for CIFAR100 dataset loaders"""
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

# Import your dataloaders
from dataset import train_loader, test_loader
TheDataset = YourCifar100Dataset(train_loader, test_loader)

# Import your DataLoader(s)
from dataset import train_loader as the_dataloader

# Import configuration variables
from config import BATCH_SIZE as the_batch_size
from config import NUM_EPOCHS as total_epochs

# --- End of modifications ---

# The grading program will use these standardized names:
# TheModel, the_trainer, the_predictor, TheDataset, the_dataloader, the_batch_size, total_epochs
print("Interface configured:")
print(f"Model: {TheModel}")
print(f"Trainer: {the_trainer}")
print(f"Predictor: {the_predictor}")
print(f"Dataset Class: {TheDataset}")
print(f"DataLoader: {the_dataloader}")
print(f"Batch Size: {the_batch_size}")
print(f"Epochs: {total_epochs}")

# Run training if this script is executed directly
if __name__ == "__main__":
    print("Starting training from interface.py...")
    try:
        trained_model = the_trainer()
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {e}")
