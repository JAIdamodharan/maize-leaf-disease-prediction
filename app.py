import streamlit as st
import numpy as np
from PIL import Image
import torch  # If using PyTorch
import tensorflow as tf  # If using TensorFlow
from model import load_model, predict
from utils import preprocess_image

def main():
    # Set page title and layout
    st.set_page_config(page_title="Image Prediction App", layout="wide")
    
    st.title("Image Prediction Application")
    st.write("Upload an image to get predictions from the model")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Initialize session state for predictions if not exists
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
        
    # Load model (only once)
    if 'model' not in st.session_state:
        with st.spinner('Loading model...'):
            st.session_state.model = load_model()
            if st.session_state.model is None:
                st.error("Error loading model. Please check your model file.")
                return
        st.success('Model loaded successfully!')
        print(f"Loaded Model Type: {type(st.session_state.model)}")  # Debugging output
    
    # If an image is uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width=300)
        
        # Get predictions button
        predict_button = st.button("Predict")
        
        if predict_button:
            with st.spinner('Processing...'):
                try:
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    print(f"Processed Image Type: {type(processed_image)}")  # Debugging output
                    
                    # Get predictions
                    predictions = predict(st.session_state.model, processed_image)
                    if predictions is None:
                        st.error("Prediction failed. Please check model compatibility.")
                        return
                    
                    st.session_state.predictions = predictions
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
        
        # Display predictions if they exist
        if st.session_state.predictions is not None:
            with col2:
                st.subheader("Prediction Results")
                
                # Display top predictions
                st.write("Top Predictions:")
                for i, (label, prob) in enumerate(st.session_state.predictions[:5]):
                    st.write(f"{i+1}. {label} ({prob:.2f}%)")
                
                # Display results as a bar chart
                chart_data = {
                    "Label": [label for label, _ in st.session_state.predictions[:5]],
                    "Probability": [prob for _, prob in st.session_state.predictions[:5]]
                }
                st.bar_chart(chart_data, x="Label", y="Probability")

if __name__ == "__main__":
    main()
