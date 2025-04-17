# predict.py
import torch
from PIL import Image
import config
from model import VGG16Model  # Import your model class name
from dataset import data_transforms  # Import transforms used during training

# Load the trained model
device = torch.device(config.DEVICE)
model = VGG16Model(num_classes=config.NUM_CLASSES).to(device)
# Load weights saved during training
model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
model.eval()  # Set model to evaluation mode

# Define the prediction function
# Rename this function as needed, e.g., my_inference_function


def cryptic_inf_f(image_path):
    """
    Takes a path to an image, preprocesses it, and returns the predicted class index.
    Adjust input/output based on specific assignment requirements (e.g., batch input).
    """
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
    # You'll need an example image path in your data/ directory or elsewhere
    # example_image = 'data/example_image.jpg'
    # prediction = cryptic_inf_f(example_image)
    # if prediction is not None:
    #     print(f"Predicted class index for {example_image}: {prediction}")
    pass  # Add test code if needed
