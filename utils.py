import numpy as np
import torch
import tensorflow as tf

def preprocess_image(image):
    """Preprocess the image before passing it to the model."""
    image = image.resize((224, 224))  # Resize to model's input size

    if isinstance(image, np.ndarray):  # If already an array, continue
        img_array = image
    else:
        img_array = np.array(image)  # Convert PIL Image to NumPy array

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    if len(img_array.shape) == 2:  # Grayscale image case
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dim

    if img_array.shape[-1] == 1:  # Convert grayscale to 3-channel
        img_array = np.repeat(img_array, 3, axis=-1)

    if isinstance(model, tf.keras.Model):  # TensorFlow
        return img_array  # TensorFlow expects (224, 224, 3)

    elif isinstance(model, torch.jit.ScriptModule):  # PyTorch
        img_array = np.transpose(img_array, (2, 0, 1))  # Convert to (3, 224, 224)
        img_array = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return img_array

    else:
        raise ValueError("Unsupported model type")
