import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.write('''
# Banana Ripeness Detection 🍌
''')
st.write("A Image Classification Web App That Detects the Ripeness Stage of Banana")

file = st.file_uploader("", type=['jpg','png'])


def predict_stage(image_data,model):
    size = (224, 224)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    preds = ""
    prediction = model.predict(data)
    if np.argmax(prediction)==0:
        st.write(f"Unripe_Stage")
    elif np.argmax(prediction)==1:
        st.write(f"Overripe_Stage")
    else :
        st.write(f"Ripe_Stage")

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model('ripeness.h5')
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        prediction = predict_stage(image, model)
