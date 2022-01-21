import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing import image

st.set_page_config(layout="wide")
cnn = tf.keras.models.load_model('cnn_model')


def load_image(image_file):
    img = Image.open(image_file)
    return img


st.subheader("Upload an Image of a Cat or Dog")
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if image_file is not None:
    # To See details
    file_details = {"filename": image_file.name, "filetype": image_file.type,
                    "filesize": image_file.size}
    st.write(file_details)

    # To View Uploaded Image
    st.image(load_image(image_file), width=250)

    test_image = image.img_to_array(load_image(image_file).resize((64, 64)))
    test_image = np.expand_dims(test_image, axis=0)

    result = cnn.predict(test_image / 255.0)

    if result[0][0] > 0.5:
        prediction = "dog"
    else:
        prediction = "cat"

    st.write("I predict this to be a: ", prediction.title())

