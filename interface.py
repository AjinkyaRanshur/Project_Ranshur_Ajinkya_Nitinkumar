# interface.py
import os
import torch
import matplotlib.pyplot as plt
import torchattacks
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # For progress bars

# --- Import your Model Class
from model import VGG16Model as TheModel

# --- Import your Training Function
from train import my_descriptively_named_train_function as the_trainer

# --- Import your Prediction Function
from predict import cryptic_inf_f as the_predictor

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


# ----- Functions -------------

def ensure_results_dir(path: str = "results") -> str:
    """
    Create the results directory if it doesn't exist.
    All attack plots will be saved here.
    """
    os.makedirs(path, exist_ok=True)
    return path


def run_fgsm_attack(model: torch.nn.Module,
                    dataloader: DataLoader,
                    device: torch.device,
                    eps: float = 0.03,
                    results_dir: str = "results") -> tuple:
    """
    FGSM Attack:
    • eps: perturbation magnitude (larger eps = stronger noise)
    Returns (clean_accuracy, adversarial_accuracy).
    """
    ensure_results_dir(results_dir)
    model.eval()

    # Measure clean accuracy
    correct_clean, total = 0, 0
    print("Evaluating clean accuracy...")
    
    for images, labels in tqdm(dataloader, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct_clean += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc_clean = correct_clean / total
    print(f"Clean accuracy: {acc_clean*100:.2f}%")

    # Generate FGSM adversarial examples
    attacker = torchattacks.FGSM(model, eps=eps)
    correct_adv = 0
    
    print(f"Running FGSM attack with eps={eps}...")
    
    for images, labels in tqdm(dataloader, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker(images, labels)
        outputs = model(adv_images)
        _, preds = outputs.max(1)
        correct_adv += (preds == labels).sum().item()
        
    acc_adv = correct_adv / total
    print(f"Adversarial accuracy: {acc_adv*100:.2f}%")

    # Plot and save
    plt.figure()
    plt.bar(["Clean", f"FGSM ε={eps}"], [acc_clean*100, acc_adv*100])
    plt.ylabel("Accuracy (%)")
    plt.title("Clean vs. FGSM Accuracy")
    plt.ylim(0, 100)
    plt.tight_layout()
    out_fp = os.path.join(results_dir, f"fgsm_acc_eps_{eps}.png")
    plt.savefig(out_fp)
    plt.close()
    print(f"[FGSM] Plot saved: {out_fp}")

    return acc_clean, acc_adv


def run_pgd_attack(model: torch.nn.Module,
                   dataloader: DataLoader,
                   device: torch.device,
                   eps: float = 0.03,
                   alpha: float = 0.007,
                   steps: int = 40,
                   results_dir: str = "results") -> tuple:
    """
    PGD Attack:
    • eps: max perturbation (L∞ bound)
    • alpha: step size each iteration
    • steps: number of iterations
    Returns (clean_accuracy, adversarial_accuracy).
    """
    ensure_results_dir(results_dir)
    model.eval()

    # Measure clean accuracy
    correct_clean, total = 0, 0
    print("Evaluating clean accuracy...")
    
    for images, labels in tqdm(dataloader, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct_clean += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc_clean = correct_clean / total
    print(f"Clean accuracy: {acc_clean*100:.2f}%")

    # Generate PGD adversarial examples
    attacker = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    correct_adv = 0
    
    print(f"Running PGD attack with eps={eps}, alpha={alpha}, steps={steps}...")
    
    for images, labels in tqdm(dataloader, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker(images, labels)
        outputs = model(adv_images)
        _, preds = outputs.max(1)
        correct_adv += (preds == labels).sum().item()
        
    acc_adv = correct_adv / total
    print(f"Adversarial accuracy: {acc_adv*100:.2f}%")

    # Plot and save
    plt.figure()
    plt.bar(["Clean", f"PGD ε={eps}"], [acc_clean*100, acc_adv*100])
    plt.ylabel("Accuracy (%)")
    plt.title(f"Clean vs. PGD Accuracy (steps={steps})")
    plt.ylim(0, 100)
    plt.tight_layout()
    out_fp = os.path.join(results_dir, f"pgd_acc_eps_{eps}_steps_{steps}.png")
    plt.savefig(out_fp)
    plt.close()
    print(f"[PGD] Plot saved: {out_fp}")

    return acc_clean, acc_adv


# Fixed the dataset handling for adversarial testing
class SimpleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def run_adversarial_tests(model_path=None):
    """Run both FGSM and PGD attacks on the model"""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = TheModel().to(device)
    if model_path:
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Test subset for faster evaluation
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
    
    # Run attacks
    print("\nRunning FGSM attack...")
    run_fgsm_attack(model, test_subset_loader, device)
    
    print("\nRunning PGD attack...")
    run_pgd_attack(model, test_subset_loader, device)
    
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
    args = parser.parse_args()
    
    if args.train:
        print("Starting training from interface.py...")
        try:
            trained_model = the_trainer()
            print("Training complete!")
        except Exception as e:
            print(f"Error during training: {e}")
    
    if args.test_adversarial:
        print("Running adversarial tests...")
        run_adversarial_tests(model_path=config.CHECKPOINT_PATH)
