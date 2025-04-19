# Config.py

# Training Hyperparameters
LEARNING_RATE = 0.0001  # Reduced learning rate
# BATCH_SIZE
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = "cuda"

# model configuration
NUM_CLASSES = 100  # CIFAR100 has 100 classes

# Dataset Configration
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224

# Normalization parameters for images(commonly used for models pretrained
# on Imagenet)
NORM_MEAN = [0.485, 0.456, 0.406]  # Fixed typo: 406 -> 0.406
NORM_STD = [0.229, 0.224, 0.225]
DATA_DIR = './data'
SHUFFLE_DATA = True
VALID_SIZE = 0.1  # Example use 10% of training data for validation

# Checkpoint path
CHECKPOINT_PATH = './checkpoints/final_weights.pth'

# You might add other parameters like weight_decay, momentum for the
# optimizer, etc.
OPTIMIZER_WEIGHT_DECAY = 0.0001  # Reduced weight decay
OPTIMIZER_MOMENTUM = 0.9

