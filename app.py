import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2


model = load_model('model.h5')


def predict_digit(image):

    image = ImageOps.grayscale(image) 
    image = image.resize((28, 28))  
    image = np.array(image) / 255.0  
    image = image.reshape(1, 28, 28, 1)  
    prediction = model.predict(image)
    return np.argmax(prediction), np.max(prediction)


st.title("Handwritten Digit Recognizer")
st.write("Draw a digit below and click 'Predict' to identify it!")


canvas_result = st.canvas(
    fill_color="white",  
    stroke_width=10,     
    stroke_color="black",  
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


if st.button("Predict"):
    if canvas_result.image_data is not None:
        
        image_data = canvas_result.image_data
        image = Image.fromarray((255 - image_data[:, :, 0]).astype('uint8'))  # Invert colors
        
     
        digit, confidence = predict_digit(image)
        st.write(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("Please draw a digit on the canvas!")
