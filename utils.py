import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess an image for the model
    
    Args:
        image: PIL Image object
        target_size: Tuple of (height, width) to resize the image to
        
    Returns:
        Preprocessed image as numpy array
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize pixel values
    img_array = img_array.astype(np.float32) / 255.0
    
    # Apply any other necessary preprocessing (depends on how the model was trained)
    # For example, if the model was trained with preprocessing from tf.keras.applications:
    # from tf.keras.applications.mobilenet_v2 import preprocess_input
    # img_array = preprocess_input(img_array * 255)
    
    return img_array