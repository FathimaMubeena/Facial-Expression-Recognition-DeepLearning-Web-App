import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import os

# Load the trained Keras model
file_path = os.path.join(os.path.dirname(__file__), 'emotion_model.keras')
model = load_model(file_path)

# Emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Streamlit title and description
st.title('Facial Emotion Recognition')
st.write('Upload a facial image to classify its emotion.')

# Upload the image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the uploaded file to a PIL image
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize the image to 48x48 (the model's expected input size)
    img = img.resize((48, 48))

    # Prepare the image for model prediction
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (grayscale)

    # Predict the emotion using the model
    prediction = model.predict(img_array)
    predicted_emotion = np.argmax(prediction)

    # Display the prediction
    st.write(f'Predicted Emotion: {emotion_labels[predicted_emotion]}')

    # Show prediction probabilities (optional)
    st.write("Prediction Probabilities:")
    for idx, prob in enumerate(prediction[0]):
        st.write(f'{emotion_labels[idx]}: {prob:.2f}')

if __name__ == '__main__':
    pass