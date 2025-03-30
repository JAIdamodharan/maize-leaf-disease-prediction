# Image Prediction Application

This Streamlit application allows users to upload images and get predictions using a pre-trained deep learning model.

## Setup Instructions 

1. Clone this repository:
```
git clone <repository-url>
cd image_prediction_app
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Update the model information:
   - Place your model file in the project directory or update the Google Drive URL in `model.py`
   - Update the `CLASS_LABELS` in `model.py` to match your model's classes

4. Run the Streamlit application:
```
streamlit run app.py
```

5. Open your browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

## Important Notes

- You need to replace `YOUR_MODEL_ID_HERE` in `model.py` with your actual model ID from Google Drive
- Update the `CLASS_LABELS` list to match your model's classes
- You may need to modify the preprocessing in `utils.py` depending on how your model was trained
- The application assumes a classification model - if you're using a different type of model, you'll need to adjust the prediction handling in `model.py`
