import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# 1. Load trained Cats vs Dogs model
# --------------------------------------------------

MODEL_PATH = "models/cats_vs_dogs_best.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# 2. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)


# --------------------------------------------------
# 3. Application title
# --------------------------------------------------

st.title("🐱 Cats vs Dogs Image Classifier")

st.write(
    "Upload an image and the trained model will predict "
    "whether it is a Cat or a Dog."
)


# --------------------------------------------------
# 4. Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    # Normalize pixels from 0-255 to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Get prediction from our trained model
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]


    # --------------------------------------------------
    # 6. Convert probability into Cat/Dog
    # --------------------------------------------------

    if prediction >= 0.5:

        label = "Dog"
        confidence = prediction

    else:

        label = "Cat"
        confidence = 1 - prediction


    # --------------------------------------------------
    # 7. Display prediction
    # --------------------------------------------------

    st.subheader(
        f"Prediction: {label}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )