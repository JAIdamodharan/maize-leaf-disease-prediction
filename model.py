import tensorflow as tf
import numpy as np
import os
import gdown

# Define class labels (replace with your actual class labels)
CLASS_LABELS = [
    "class_0", "class_1", "class_2", "class_3", "class_4", 
    "class_5", "class_6", "class_7", "class_8", "class_9"
]

def download_model_if_needed(model_path='model.h5', gdrive_url=None):
    """Download the model from Google Drive if it doesn't exist locally"""
    if not os.path.exists(model_path) and gdrive_url:
        print(f"Downloading model from {gdrive_url}...")
        gdown.download(gdrive_url, model_path, quiet=False)
    
    return model_path

def load_model():
    """Load the TensorFlow model
    
    Returns:
        A TensorFlow model
    """
    # Extract model URL from your Colab notebook
    gdrive_url = "https://drive.google.com/uc?id=YOUR_MODEL_ID_HERE"
    
    # Download model if needed
    model_path = download_model_if_needed(gdrive_url=gdrive_url)
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def predict(model, image):
    """Make predictions using the loaded model
    
    Args:
        model: TensorFlow model
        image: Preprocessed image as numpy array
        
    Returns:
        List of (class_name, probability) tuples sorted by probability
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)
    
    # Process the predictions
    if predictions.shape[1] == len(CLASS_LABELS):
        # For classification model
        # Get predicted probabilities and corresponding class labels
        results = []
        for i, prob in enumerate(predictions[0]):
            results.append((CLASS_LABELS[i], float(prob * 100)))
        
        # Sort results by probability in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    else:
        # Handle other types of models as needed
        raise NotImplementedError("Prediction processing for this model type not implemented")
