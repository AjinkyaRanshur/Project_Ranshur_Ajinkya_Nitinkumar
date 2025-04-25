# pcoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class PCoder(nn.Module):
    """
    Predictive coding module that wraps around a model layer.
    This implementation uses representational predictive coding where a representation
    is iteratively refined based on feedback and feedforward signals.
    """
    def __init__(self, layer_name, predictor_module, has_feedback=True, random_init=False):
        super().__init__()
        self.layer_name = layer_name
        self.predictor = predictor_module  # The prediction network
        self.has_feedback = has_feedback   # Whether this layer receives feedback
        self.random_init = random_init     # Whether to initialize randomly
        self.rep = None                    # Current representation
        self.grd = None                    # Gradient of representation

    def forward(self, ff_input, fb_input=None, target=None, 
                build_graph=False, ffm=0.3, fbm=0.3, pcm=0.01):
        """
        Perform one step of predictive coding.
        
        Args:
            ff_input: Feedforward input (typically from the layer we're attached to)
            fb_input: Feedback input (typically from the next layer)
            target: Target representation (typically from the previous layer)
            build_graph: Whether to keep the computation graph for backprop
            ffm: Feedforward momentum (how much weight to give to the feedforward signal)
            fbm: Feedback momentum (how much weight to give to the feedback signal)
            pcm: Predictive coding momentum (how much weight to give to the prediction error)
            
        Returns:
            rep: The current representation
            pred: The prediction from this representation
        """
        # Get device from input
        device = ff_input.device
        
        # Initialize or update representation
        if self.rep is None:
            # First pass - initialize with feedforward or random
            self.rep = torch.randn_like(ff_input) if self.random_init else ff_input.clone()
            self.grd = torch.zeros_like(self.rep)
        else:
            if self.has_feedback and fb_input is not None:
                # If spatial dims don't match, upsample fb_input to match ff_input
                if fb_input.shape[2:] != ff_input.shape[2:]:
                    fb_input = F.interpolate(
                        fb_input, size=ff_input.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                # Update representation with feedforward and feedback signals
                self.rep = (
                    ffm * ff_input +
                    fbm * fb_input +
                    (1 - ffm - fbm) * self.rep -
                    pcm * self.grd
                )
    
        # If we have a target, compute prediction error and gradient
        if target is not None:
            with torch.enable_grad():
                if not self.rep.requires_grad:
                    self.rep.requires_grad_(True)
                    
                # Make prediction from current representation
                pred = self.predictor(self.rep)
                
                # If spatial dims don't match, upsample pred to match target
                if pred.shape[2:] != target.shape[2:]:
                    pred = F.interpolate(
                        pred, size=target.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # Compute prediction error and its gradient
                err = F.mse_loss(pred, target)
                self.grd = torch.autograd.grad(err, self.rep, retain_graph=build_graph)[0]
        else:
            # No target, just use predictor to generate output
            pred = self.predictor(self.rep)
            
        # Detach if not building computation graph
        if not build_graph:
            if self.grd is not None:
                self.grd = self.grd.detach()
            self.rep = self.rep.detach()
            pred = pred.detach()
    
        return self.rep, pred


class PredictiveCoder:
    """
    Manages the integration of predictive coding into a model.
    """
    def __init__(self, model, pc_configs, device=None):
        """
        Initialize the predictive coder with a model and configuration.
        
        Args:
            model: The PyTorch model to wrap
            pc_configs: List of dictionaries containing predictive coding configuration
            device: The device to use (automatically detected if None)
        """
        self.model = model
        self.pc_configs = pc_configs
        self.device = device or next(model.parameters()).device
        self.pcoders = []
        self.hook_handles = []
        self.activations = {}
        
        # Store original forward method
        self.original_forward = model.forward
        
        print(f"Initializing predictive coding with {len(pc_configs)} layers on {self.device}...")

    def _get_activation(self, name):
        """Create a hook function to record activations for a specific layer."""
        def hook(module, input, output):
            self.activations[name] = output
        return hook
    
    def integrate(self):
        """
        Integrate predictive coding modules into the model.
        Returns the modified model with PCoder modules.
        """
        # Register hooks to capture layer outputs
        for idx, pc_config in enumerate(self.pc_configs):
            layer_name = pc_config['layer_name']
            module_name = pc_config['from_module']
            
            # Get the module
            if not hasattr(self.model, module_name):
                print(f"Warning: Module {module_name} not found in model. Skipping.")
                continue
                
            module = getattr(self.model, module_name)
            
            # Register forward hook
            handle = module.register_forward_hook(self._get_activation(module_name))
            self.hook_handles.append(handle)
            
            try:
                # Create predictor module from string
                predictor = eval(pc_config['predictor'])
                predictor = predictor.to(self.device)
                
                has_feedback = pc_config['hyperparameters']['feedback'] > 0
                
                # Create PCoder
                pcoder = PCoder(
                    layer_name=layer_name,
                    predictor_module=predictor,
                    has_feedback=has_feedback,
                    random_init=False
                )
                
                # Move PCoder to device
                pcoder = pcoder.to(self.device)
                
                # Add PCoder to model
                setattr(self.model, f"pcoder_{idx}", pcoder)
                self.pcoders.append((idx, pcoder, pc_config))
                print(f"Added PCoder {idx} for layer {module_name}")
                
            except Exception as e:
                print(f"Error creating PCoder for {module_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Replace forward method
        self.model.forward = self._forward_with_pc
        
        # Add cleanup method
        self.model.cleanup_pc_hooks = self.cleanup_hooks
        
        # Add toggle method
        self.model.toggle_pc_mode = self.toggle_pc_mode
        
        return self.model
    
    def _forward_with_pc(self, x):
        """Forward pass with predictive coding."""
        # First pass: normal forward to collect activations
        output = self.original_forward(x)
        
        # Process each PCoder in reverse order (top-down feedback)
        for idx in range(len(self.pcoders) - 1, -1, -1):
            pcoder_idx, pcoder, pc_config = self.pcoders[idx]
            module_name = pc_config['from_module']
            
            # Skip if we don't have the activation
            if module_name not in self.activations:
                continue
                
            # Get feedforward input
            ff_input = self.activations[module_name]
            
            # Get feedback input (from next PCoder if available)
            fb_input = None
            if idx < len(self.pcoders) - 1:
                next_pcoder = self.pcoders[idx + 1][1]
                if next_pcoder.rep is not None:
                    fb_input = next_pcoder.rep
            
            # Get target (from previous PCoder if available)
            target = None
            if idx > 0:
                prev_pcoder = self.pcoders[idx - 1][1]
                if prev_pcoder.rep is not None:
                    target = prev_pcoder.rep
            
            # Apply predictive coding step
            hp = pc_config['hyperparameters']
            rep, _ = pcoder(
                ff_input=ff_input,
                fb_input=fb_input,
                target=target,
                build_graph=True,
                ffm=hp['feedforward'],
                fbm=hp['feedback'],
                pcm=hp['pc']
            )
            
            # Replace the activation with the predictive coding representation
            self.activations[module_name] = rep
        
        return output
    
    def cleanup_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def toggle_pc_mode(self, enabled=True):
        """Toggle predictive coding mode."""
        if enabled:
            self.model.forward = self._forward_with_pc
            print("Predictive coding mode enabled")
        else:
            self.model.forward = self.original_forward
            # Reset PCoder states
            for _, pcoder, _ in self.pcoders:
                pcoder.rep = None
                pcoder.grd = None
            print("Predictive coding mode disabled")


def run_fgsm_attack(model, dataloader, device, eps=0.03, results_dir="./plots/results"):
    """
    FGSM Attack test.
    
    Args:
        model: The model to attack
        dataloader: Dataloader for test data
        device: Device to run the attack on
        eps: Perturbation magnitude
        results_dir: Directory to save results
        
    Returns:
        (clean_accuracy, adversarial_accuracy) tuple
    """
    import torchattacks
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
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


def run_pgd_attack(model, dataloader, device, eps=0.03, alpha=0.007, steps=40, results_dir="./plots/results"):
    """
    PGD Attack test.
    
    Args:
        model: The model to attack
        dataloader: Dataloader for test data
        device: Device to run the attack on
        eps: Max perturbation magnitude
        alpha: Step size per iteration
        steps: Number of iterations
        results_dir: Directory to save results
        
    Returns:
        (clean_accuracy, adversarial_accuracy) tuple
    """
    import torchattacks
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
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


def create_comparison_plots(results, results_dir="./plots/results/comparison"):
    """
    Create comparison plots between models with and without predictive coding.
    
    Args:
        results: Dictionary containing results
        results_dir: Directory to save plots
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # FGSM comparison
    if 'no_pc_fgsm' in results and 'with_pc_fgsm' in results:
        plt.figure(figsize=(10, 6))
        labels = ['Clean (No PC)', 'FGSM (No PC)', 'Clean (With PC)', 'FGSM (With PC)']
        values = [
            results['no_pc_fgsm'][0] * 100,  # Clean accuracy without PC
            results['no_pc_fgsm'][1] * 100,  # FGSM accuracy without PC
            results['with_pc_fgsm'][0] * 100, # Clean accuracy with PC
            results['with_pc_fgsm'][1] * 100  # FGSM accuracy with PC
        ]
        
        plt.bar(labels, values, color=['blue', 'red', 'green', 'orange'])
        plt.ylabel('Accuracy (%)')
        plt.title('FGSM Attack: With vs Without Predictive Coding')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'fgsm_comparison.png'))
        plt.close()
        print("FGSM comparison plot saved")
        
    # PGD comparison
    if 'no_pc_pgd' in results and 'with_pc_pgd' in results:
        plt.figure(figsize=(10, 6))
        labels = ['Clean (No PC)', 'PGD (No PC)', 'Clean (With PC)', 'PGD (With PC)']
        values = [
            results['no_pc_pgd'][0] * 100,   # Clean accuracy without PC
            results['no_pc_pgd'][1] * 100,   # PGD accuracy without PC
            results['with_pc_pgd'][0] * 100,  # Clean accuracy with PC
            results['with_pc_pgd'][1] * 100   # PGD accuracy with PC
        ]
        
        plt.bar(labels, values, color=['blue', 'red', 'green', 'orange'])
        plt.ylabel('Accuracy (%)')
        plt.title('PGD Attack: With vs Without Predictive Coding')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'pgd_comparison.png'))
        plt.close()
        print("PGD comparison plot saved")
        
    # Combined resistance plot
    if all(k in results for k in ['no_pc_fgsm', 'with_pc_fgsm', 'no_pc_pgd', 'with_pc_pgd']):
        plt.figure(figsize=(10, 6))
        
        # Calculate resistance as: adversarial_acc / clean_acc * 100%
        resistance_values = [
            results['no_pc_fgsm'][1] / results['no_pc_fgsm'][0] * 100,  # FGSM resistance without PC
            results['with_pc_fgsm'][1] / results['with_pc_fgsm'][0] * 100,  # FGSM resistance with PC
            results['no_pc_pgd'][1] / results['no_pc_pgd'][0] * 100,   # PGD resistance without PC
            results['with_pc_pgd'][1] / results['with_pc_pgd'][0] * 100    # PGD resistance with PC
        ]
        
        labels = ['FGSM (No PC)', 'FGSM (With PC)', 'PGD (No PC)', 'PGD (With PC)']
        plt.bar(labels, resistance_values, color=['red', 'orange', 'purple', 'green'])
        plt.ylabel('Adversarial Resistance (%)')
        plt.title('Adversarial Resistance: % of Clean Accuracy Retained')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'resistance_comparison.png'))
        plt.close()
        print("Adversarial resistance comparison plot saved")
        
    # Accuracy drop under attack
    if all(k in results for k in ['no_pc_fgsm', 'with_pc_fgsm', 'no_pc_pgd', 'with_pc_pgd']):
        plt.figure(figsize=(10, 6))
        
        # Calculate accuracy drop: clean_acc - adversarial_acc
        drop_values = [
            (results['no_pc_fgsm'][0] - results['no_pc_fgsm'][1]) * 100,  # FGSM drop without PC
            (results['with_pc_fgsm'][0] - results['with_pc_fgsm'][1]) * 100,  # FGSM drop with PC
            (results['no_pc_pgd'][0] - results['no_pc_pgd'][1]) * 100,   # PGD drop without PC
            (results['with_pc_pgd'][0] - results['with_pc_pgd'][1]) * 100    # PGD drop with PC
        ]
        
        labels = ['FGSM (No PC)', 'FGSM (With PC)', 'PGD (No PC)', 'PGD (With PC)']
        plt.bar(labels, drop_values, color=['darkred', 'red', 'darkblue', 'blue'])
        plt.ylabel('Accuracy Drop (%)')
        plt.title('Performance Drop Under Adversarial Attack')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'accuracy_drop_comparison.png'))
        plt.close()
        print("Accuracy drop comparison plot saved")
