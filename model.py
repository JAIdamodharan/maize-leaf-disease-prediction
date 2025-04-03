import torch  # If using PyTorch
import tensorflow as tf  # If using TensorFlow
import numpy as np

def load_model():
    try:
        model_path = "best_model.keras"  # Change based on your model type
        model = None  # Ensure model is defined
        
        if model_path.endswith(".keras"):  # TensorFlow Model
            model = tf.keras.models.load_model(model_path)
            print("Loaded model type:", type(model))
        
        elif model_path.endswith(".pt") or model_path.endswith(".pth"):  # PyTorch Model
            model = torch.jit.load(model_path)
            model.eval()
            print("Loaded model type:", type(model))
        
        else:
            print("Unsupported model format")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict(model, image):
    if model is None:
        raise ValueError("Model is not loaded properly.")

    try:
        if isinstance(model, tf.keras.Model):  # TensorFlow Model
            predictions = model.predict(np.expand_dims(image, axis=0))[0]

        elif isinstance(model, torch.jit.ScriptModule):  # PyTorch Model
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
            outputs = model(image_tensor)
            predictions = torch.softmax(outputs, dim=1).detach().numpy()[0]

        else:
            raise NotImplementedError("Prediction processing for this model type not implemented")
        
        # Convert predictions to label-probability pairs
        class_labels = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]  # Modify based on your classes
        result = sorted(zip(class_labels, predictions), key=lambda x: x[1], reverse=True)
        
        return result
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
