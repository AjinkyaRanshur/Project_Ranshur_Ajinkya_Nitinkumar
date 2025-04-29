#interface.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # For progress bars

# --- Import your Model Class
from model import VGG16Model as TheModel

# --- Import your Training Function
from train import my_descriptively_named_train_function as the_trainer

# --- Import your Prediction Function
from predict import cryptic_inf_f as the_predictor

# --- Import Predictive Coding Module
from pcoder import PredictiveCoder, run_fgsm_attack, run_pgd_attack, create_comparison_plots

# --- Import your Dataset Class
# Creating a simple wrapper class for the datasets
class YourCifar100Dataset:
    """Wrapper class for CIFAR100 dataset loaders"""
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

# --- Import your dataloaders
from dataset import train_loader, test_loader
TheDataset = YourCifar100Dataset(train_loader, test_loader)

# --- Import your DataLoader(s)
from dataset import train_loader as the_dataloader

# --- Import configuration variables
from config import BATCH_SIZE as the_batch_size
from config import NUM_EPOCHS as total_epochs
from config import DEVICE
from config import CHECKPOINT_PATH
from config import PREDICTIVE_CODERS


def ensure_results_dir(path: str = "./plots/results") -> str:
    """
    Create the results directory if it doesn't exist.
    All attack plots will be saved here.
    """
    os.makedirs(path, exist_ok=True)
    return path


# Fixed the dataset handling for adversarial testing
class SimpleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def run_adversarial_tests(model_path=None, use_pc=False):
    """Run both FGSM and PGD attacks on the model with and without predictive coding"""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = TheModel().to(device)
    if model_path:
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare test subset
    images_list = []
    labels_list = []
    test_count = 0
    max_test_samples = 100  # Limit number of test samples for speed
    
    print("Creating test subset...")
    for images, labels in tqdm(test_loader, desc="Preparing test subset"):
        images_list.append(images)
        labels_list.append(labels)
        test_count += images.size(0)
        if test_count >= max_test_samples:
            break
    
    # Concatenate all batches
    all_images = torch.cat(images_list, 0)
    all_labels = torch.cat(labels_list, 0)
    
    # Limit to max_test_samples
    all_images = all_images[:max_test_samples]
    all_labels = all_labels[:max_test_samples]
    
    print(f"Using {len(all_images)} samples for adversarial testing")
    
    # Create proper dataloader for the subset
    test_subset = SimpleDataset(all_images, all_labels)
    test_subset_loader = DataLoader(
        test_subset, 
        batch_size=the_batch_size,
        shuffle=False
    )
    
    # Store the results for comparison plotting
    results = {}
    
    # First, run attacks without PC
    print("\nRunning attacks without predictive coding...")
    
    print("\nRunning FGSM attack (without PC)...")
    acc_clean_no_pc, acc_fgsm_no_pc = run_fgsm_attack(
        model, test_subset_loader, device, 
        results_dir="./plots/results/no_pc"
    )
    results['no_pc_fgsm'] = (acc_clean_no_pc, acc_fgsm_no_pc)
    
    print("\nRunning PGD attack (without PC)...")
    acc_clean_no_pc, acc_pgd_no_pc = run_pgd_attack(
        model, test_subset_loader, device,
        results_dir="./plots/results/no_pc"
    )
    results['no_pc_pgd'] = (acc_clean_no_pc, acc_pgd_no_pc)
    
    # Now, if requested, apply predictive coding and run attacks again
    if use_pc:
        print("\nApplying predictive coding to model...")
        # Use our new PredictiveCoder class
        pc = PredictiveCoder(model, PREDICTIVE_CODERS, device)
        model = pc.integrate()
        
        print("\nRunning FGSM attack (with PC)...")
        acc_clean_pc, acc_fgsm_pc = run_fgsm_attack(
            model, test_subset_loader, device,
            results_dir="./plots/results/with_pc"
        )
        results['with_pc_fgsm'] = (acc_clean_pc, acc_fgsm_pc)
        
        print("\nRunning PGD attack (with PC)...")
        acc_clean_pc, acc_pgd_pc = run_pgd_attack(
            model, test_subset_loader, device,
            results_dir="./plots/results/with_pc"
        )
        results['with_pc_pgd'] = (acc_clean_pc, acc_pgd_pc)
        
        # Clean up hooks
        model.cleanup_pc_hooks()
        
        # Create comparison plots
        create_comparison_plots(results)
    
    print("\nAdversarial testing complete. Check the 'results' directory for plots.")


#-----------------------------

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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training and adversarial testing")
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Train the model before testing"
    )
    parser.add_argument(
        "--test-adversarial",
        action="store_true",
        help="Run adversarial attacks on the model"
    )	
    parser.add_argument(
        "--predictive", 
        action="store_true",
        help="Enable predictive coding layers"
    )
    args = parser.parse_args()
    
    if args.train:
        print("Starting training from interface.py...")
        try:
            trained_model = the_trainer()
            print("Training complete!")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
    
    if args.test_adversarial:
        print("Running adversarial tests...")
        run_adversarial_tests(model_path=CHECKPOINT_PATH, use_pc=args.predictive)
