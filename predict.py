# predict.py
import torch
from PIL import Image
import config
from model import VGG16Model  # Import your model class name
from dataset import data_transforms  # Import transforms used during training

# Define the prediction function
def cryptic_inf_f(image_path, model_path=config.CHECKPOINT_PATH):
    """
    Takes a path to an image, preprocesses it, and returns the predicted class index.
    """
    device = torch.device(config.DEVICE)
    
    # Load the model
    model = VGG16Model(num_classes=config.NUM_CLASSES).to(device)
    
    # Try to load weights if they exist
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode
    except FileNotFoundError:
        print(f"Warning: Model weights not found at {model_path}. Using untrained model.")
        model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB')
        # Apply the same transformations as the test set
        input_tensor = data_transforms['test'](image)
        # Add batch dimension (B, C, H, W) -> (1, C, H, W)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            # Get predicted class index
            _, predicted_idx = torch.max(output.data, 1)
            return predicted_idx.item()  # Return the class index

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Example usage (optional)
if __name__ == '__main__':
    # Test code if needed
    pass
