import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("model.h5")

class_names = [
"Apple_scab",
"Tomato_Early_blight",
"Potato_Early_blight",
"Healthy"
]

st.title("🌿 Plant Disease Detection AI")

uploaded_file = st.file_uploader(
"Upload leaf image",
type=["jpg","png"]
)

def segment_disease(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25,40,40])
    upper_green = np.array([90,255,255])

    leaf_mask = cv2.inRange(hsv,lower_green,upper_green)

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    _,disease_mask = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)

    mask = cv2.bitwise_and(disease_mask,disease_mask,mask=leaf_mask)

    return mask


if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    img_resize = cv2.resize(img,(128,128))
    img_resize = img_resize/255.0

    pred = model.predict(np.expand_dims(img_resize,axis=0))
    label = class_names[np.argmax(pred)]

    mask = segment_disease(img)

    st.image(img,caption="Original Image")

    st.write("Prediction:",label)

    st.image(mask,caption="Disease Region")
