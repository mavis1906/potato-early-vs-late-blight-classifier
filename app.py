import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/potato_classifier.keras")

class_names= ["early blight", "late blight"]

st.title("potato leaf disease classifier")
st.write("upload a potato leaf image to classify it.")

uploaded_file= st.file_uploader(
    "choose a potato leaf image",
    type=["jpg", "jpg", "png"]
)
if uploaded_file is not None:
    image= Image.open(uploaded_file).convert("RGB")
    st.image(image, caption= "uploaded image", use_container_width= True)

    image = image.resize((224,224))
    image_array= np.array(image) / 255.0
    image_array= np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    if prediction[0][0] < 0.5:
        st.success("prediction: early blight")
        st.write(f"cofidence: {(1-prediction[0][0])*100:.2f}%")
    else:
        st.success("prediction: late blight")
        st.write(f"confident:{prediction[0][0]*100:.2f}%")