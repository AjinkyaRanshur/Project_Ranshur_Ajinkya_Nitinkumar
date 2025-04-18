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


# ----- Functions-------------

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
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct_clean += (preds == labels).sum().item()
        total += labels.size(0)
    acc_clean = correct_clean / total

    # Generate FGSM adversarial examples
    attacker = torchattacks.FGSM(model, eps=eps)
    correct_adv = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker(images, labels)
        outputs = model(adv_images)
        _, preds = outputs.max(1)
        correct_adv += (preds == labels).sum().item()
    acc_adv = correct_adv / total

    # Plot and save
    plt.figure()
    plt.bar(["Clean", f"FGSM ε={eps}"], [acc_clean, acc_adv])
    plt.ylabel("Accuracy")
    plt.title("Clean vs. FGSM Accuracy")
    plt.ylim(0, 1)
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
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct_clean += (preds == labels).sum().item()
        total += labels.size(0)
    acc_clean = correct_clean / total

    # Generate PGD adversarial examples
    attacker = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    correct_adv = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker(images, labels)
        outputs = model(adv_images)
        _, preds = outputs.max(1)
        correct_adv += (preds == labels).sum().item()
    acc_adv = correct_adv / total

    # Plot and save
    plt.figure()
    plt.bar(["Clean", f"PGD ε={eps}"], [acc_clean, acc_adv])
    plt.ylabel("Accuracy")
    plt.title(f"Clean vs. PGD Accuracy (steps={steps})")
    plt.ylim(0, 1)
    plt.tight_layout()
    out_fp = os.path.join(results_dir, f"pgd_acc_eps_{eps}_steps_{steps}.png")
    plt.savefig(out_fp)
    plt.close()
    print(f"[PGD] Plot saved: {out_fp}")

    return acc_clean, acc_adv

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
    print("Starting training from interface.py...")
    try:
        trained_model = the_trainer()
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {e}")
