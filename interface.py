# interface.py

# --- IMPORTANT: Modify the names on the left (before 'import') ---
# --- and the names on the right (after 'import' and before 'as') ---
# --- to match YOUR actual class/function/variable names ---

# Import your Model Class
# Replace VGG16Model with your model class name
from model import VGG16Model as TheModel

# Import your Training Function
# Replace with your train function name
from train import my_descriptively_named_train_function as the_trainer

# Import your Prediction Function
# Replace with your predict function name
from predict import cryptic_inf_f as the_predictor

# Import your Dataset Class
# If you created a custom Dataset class:
# from dataset import YourCifar100Dataset as TheDataset # Replace YourCifar100Dataset with your dataset class name
# If you are only expected to provide the DataLoader and not a specific Dataset class,
# you might need clarification or adjust this line based on assignment specifics.
# For now, let's assume a placeholder might be needed or you have a
# wrapper class:
from dataset import YourCifar100Dataset as TheDataset  # Replace if necessary

# Import your DataLoader(s)
# Choose the correct loader (train/test/validation) expected by the grader if only one is needed,
# or provide the training loader if that's the primary one.
# Replace your_train_loader with your main dataloader variable name
from dataset import your_train_loader as the_dataloader

# Import configuration variables
# Ensure BATCH_SIZE is in config.py
from config import BATCH_SIZE as the_batch_size
# Ensure NUM_EPOCHS is in config.py
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
