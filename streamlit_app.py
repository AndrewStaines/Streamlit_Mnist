# Import necessary libraries
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your model
model = load_model("MNIST_model")

# Streamlit app
st.title('MNIST Image Prediction App')

# Sidebar with user input for prediction
st.sidebar.header('User Input')

# File uploader for the user to upload an image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for prediction
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Display prediction
    st.subheader('Prediction')
    st.write(f'The model predicts: {predicted_class}')

    # Display the probabilities for each class
    st.subheader('Prediction Probabilities')
    st.write(predictions)

    # Optional: Display the uploaded image
    st.subheader('Uploaded Image')
    st.image(img, caption='Uploaded Image.', use_column_width=True)
