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
NORM_MEAN = [0.485, 0.456, 0.406]
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

# Predictive coding configuration adjusted for your VGG16 model structure
# Each predictive coder maps to a specific layer in your model architecture
PREDICTIVE_CODERS = [
    {
        # Block 1 - corresponds to features[3] (after conv1_2)
        'layer_name': 'layer1',
        'from_module': 'bn1_2',  # Layer whose output we want to encode
        'predictor': 'torch.nn.ConvTranspose2d(64, 3, kernel_size=5, padding=2)',  # Prediction network
        'hyperparameters': {
            'feedforward': 0.2,
            'feedback': 0.05,
            'pc': 0.01,
        }
    },
    {
        # Block 2 - corresponds to features[8] (after conv2_2)
        'layer_name': 'layer2',
        'from_module': 'bn2_2',
        'predictor': 'torch.nn.Sequential(torch.nn.ConvTranspose2d(128, 64, kernel_size=10, stride=2, padding=4), torch.nn.ReLU(inplace=True))',
        'hyperparameters': {
            'feedforward': 0.4,
            'feedback': 0.1,
            'pc': 0.01,
        }
    },
    {
        # Block 3 - corresponds to features[15] (after conv3_3)
        'layer_name': 'layer3',
        'from_module': 'bn3_3',
        'predictor': 'torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 128, kernel_size=14, stride=2, padding=6), torch.nn.ReLU(inplace=True))',
        'hyperparameters': {
            'feedforward': 0.4,
            'feedback': 0.1,
            'pc': 0.01,
        }
    },
    {
        # Block 4 - corresponds to features[22] (after conv4_3)
        'layer_name': 'layer4',
        'from_module': 'bn4_3',
        'predictor': 'torch.nn.Sequential(torch.nn.ConvTranspose2d(512, 256, kernel_size=14, stride=2, padding=6), torch.nn.ReLU(inplace=True))',
        'hyperparameters': {
            'feedforward': 0.5,
            'feedback': 0.1,
            'pc': 0.01,
        }
    },
    {
        # Block 5 - corresponds to features[29] (after conv5_3)
        'layer_name': 'layer5',
        'from_module': 'bn5_3',
        'predictor': 'torch.nn.Sequential(torch.nn.ConvTranspose2d(512, 512, kernel_size=14, stride=2, padding=6), torch.nn.ReLU(inplace=True))',
        'hyperparameters': {
            'feedforward': 0.6,
            'feedback': 0.0,
            'pc': 0.01,
        }
    }
]
