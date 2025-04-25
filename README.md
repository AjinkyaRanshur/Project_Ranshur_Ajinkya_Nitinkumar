# Project_Ranshur_Ajinkya_Nitinkumar

## Adversarially Robust VGG16 with Predictive Coding for CIFAR-100

This repository contains the implementation of a custom VGG16 architecture trained on the CIFAR-100 dataset with predictive coding layers to improve robustness against adversarial attacks. The project was completed as a part of the Image and Video Processing course by Chaitanya Guttikar.

## Project Description

### Overview
This project explores the application of predictive coding principles to enhance the robustness of convolutional neural networks (specifically VGG16) against adversarial attacks. We implement representational predictive coding layers that iteratively refine representations based on feedback and feedforward signals, creating a more robust feature hierarchy that is less susceptible to adversarial perturbations.

### Model Architecture
We implement a custom VGG16 architecture with batch normalization throughout all convolutional layers. The model consists of:
- 5 blocks of convolutional layers with increasing filter counts (64, 128, 256, 512, 512)
- Each block uses 3x3 convolutions followed by batch normalization and ReLU activation
- Max pooling between blocks
- Fully connected layers with dropout for classification (4096 → 4096 → 100 classes)

### Predictive Coding Framework
Our predictive coding implementation is inspired by the hierarchical predictive coding theory from neuroscience (Rao & Ballard, 1999; Friston, 2010). We integrate predictive coders between specific network layers that:
1. Generate predictions about lower-level activations
2. Update representations based on prediction errors
3. Create a bidirectional information flow with both feedforward and feedback pathways

Each predictive coder has:
- A prediction network (implemented as transposed convolutions)
- Configurable hyperparameters for balancing feedforward/feedback signals
- A mechanism to update representations based on prediction errors

### Adversarial Attack Testing
We test the model against two common adversarial attacks:
1. **Fast Gradient Sign Method (FGSM)** (Goodfellow et al., 2014): A single-step attack that perturbs inputs in the direction of the gradient of the loss with respect to the input
2. **Projected Gradient Descent (PGD)** (Madry et al., 2017): A stronger, multi-step iterative attack that creates adversarial examples within a specified perturbation budget

Our framework measures and compares model performance with and without predictive coding under these attacks.

## Dataset

We use the CIFAR-100 dataset, which contains 60,000 32x32 color images in 100 different classes. Images are resized to 224x224 for compatibility with the VGG architecture.

### Dataset Structure
The code expects the following directory structure:
```
data/
├── train/
│   ├── class_0/
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```

### Download Instructions
1. Download the CIFAR-100 dataset:
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz
```

2. Convert the dataset to the expected format:
```python
import pickle
import numpy as np
import os
from PIL import Image

# Load the CIFAR-100 data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Create directories
for split in ['train', 'test']:
    for i in range(100):
        os.makedirs(f"data/{split}/class_{i}", exist_ok=True)

# Process training data
train_data = unpickle('cifar-100-python/train')
for i, (image, label) in enumerate(zip(train_data[b'data'], train_data[b'fine_labels'])):
    img = image.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    img.save(f"data/train/class_{label}/img_{i}.png")

# Process test data
test_data = unpickle('cifar-100-python/test')
for i, (image, label) in enumerate(zip(test_data[b'data'], test_data[b'fine_labels'])):
    img = image.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    img.save(f"data/test/class_{label}/img_{i}.png")
```

## Installation Instructions

### Requirements
- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- tqdm
- torchattacks (for adversarial attack testing)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/username/Project_Ranshur_Ajinkya_Nitinkumar.git
cd Project_Ranshur_Ajinkya_Nitinkumar
```

2. Install the required packages:
```bash
pip install torch torchvision matplotlib numpy tqdm torchattacks
```

3. Prepare the dataset as described in the Dataset section.

## Usage

### Training
To train the model from scratch:
```bash
python train.py
```

To force retraining even if a checkpoint exists:
```bash
python train.py --force-retrain
```

### Complete Pipeline (Train + Adversarial Testing)
```bash
python interface.py --train --predictive --test-adversarial
```

This command will:
1. Train the VGG16 model on CIFAR-100
2. Apply predictive coding layers
3. Test against FGSM and PGD adversarial attacks
4. Generate comparison plots

### Testing CUDA Availability
```bash
python checkcuda.py
```

### Running on HPC with SLURM
Submit the job using:
```bash
sbatch run.sh
```

## Model Pipeline

1. **Data Loading**: CIFAR-100 images are loaded, resized to 224x224, and normalized.
2. **Training**: The VGG16 model is trained using Adam optimizer with cosine annealing learning rate scheduling.
3. **Predictive Coding Integration**: 
   - Predictive coders are attached to specific layers in the VGG16 model
   - Each coder defines a prediction network that maps higher-level features to lower-level ones
   - During inference, representations are iteratively refined using both feedback and feedforward signals
4. **Adversarial Testing**:
   - FGSM attack: Single-step perturbation with epsilon=0.03
   - PGD attack: Multi-step iterative attack with epsilon=0.03, alpha=0.007, steps=40
   - Performance is measured with and without predictive coding
5. **Results Visualization**: Comparative plots are generated showing the model's robustness against attacks.

## Key Papers on Predictive Coding

1. Rao, R.P. and Ballard, D.H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79-87.

2. Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.

3. Lotter, W., Kreiman, G., & Cox, D. (2017). Deep predictive coding networks for video prediction and unsupervised learning. ICLR 2017.

4. Wen, H., Han, K., Shi, J., Zhang, Y., Culurciello, E., & Liu, Z. (2018). Deep predictive coding network for object recognition. ICML 2018.

5. Chalasani, R., & Principe, J. C. (2013). Deep predictive coding networks. arXiv preprint arXiv:1301.3541.

6. Salvatori, T., Song, Y., Hong, Y., Sha, L., & Frieder, S. (2021). Adversarial Robustness of Deep Predictive Coding Networks. Neural Networks, 142, 433-443.

## Repository Structure

```
/
├── __pycache__/                 # Python bytecode cache  
├── data/                        # Expected CIFAR‑100 train/test folders  
├── plots/results/               # Saved adversarial plots  
├── cmd_list                     # Single-line job command: interface.py with train+attack   
├── cmd_list_test                # Command to check CUDA availability 
├── checkcuda.py                 # Prints torch.cuda.is_available()    
├── config.py                    # Hyperparameters & paths 
├── dataset.py                   # DataLoader setup for CIFAR‑100   
├── interface.py                 # CLI tying model/train/predict/adversarial tests  
├── model.py                     # Custom VGG‑16 architecture implementation   
├── pcoder.py                    # Predictive coding implementation
├── predict.py                   # Single‑image inference function  
├── run.sh                       # SLURM submission script  
├── train.py                     # Training loop, checkpointing & plotting   
└── README.md                    # Project description
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
