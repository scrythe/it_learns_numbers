import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from cropper_component import cropper_component
import numpy as np
import mimetypes
import pickle
import matplotlib.pyplot as plt
import cv2


def load_trained_network():
    with open("network.pickle", "rb") as file:
        network = pickle.load(file)
    return network


def save_image(image):
    fig, _ = plt.subplots()
    plt.imshow(image.reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    st.pyplot(fig)


def save_propability_graph(prediction):
    fig, _ = plt.subplots()
    digits = np.arange(10)
    plt.bar(digits, prediction)

    for i, prob in enumerate(prediction):
        plt.text(i, prob * 1.2, f"{prob:.1e}", ha="center", va="bottom", fontsize=10)

    plt.xticks(digits)
    plt.ylim(0, 1)
    plt.xlabel("Ziffer")
    plt.ylabel("Wahrscheinlichkeit")
    plt.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig)


def convert_to_image_src(uploaded_image):
    image = Image.open(uploaded_image)
    buffered = BytesIO()
    image.save(buffered, format=image.format)
    base64_data = base64.b64encode(buffered.getvalue()).decode()
    mime, _ = mimetypes.guess_type(uploaded_image.name)
    if not mime:
        mime = "image/png"
    image_src = f"data:{mime};base64,{base64_data}"
    return image_src


def convert_to_image(image_src):
    _, base64_data = image_src.split(",", 1)
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data)).convert("L")
    return image


def preprocess_image(image):
    resized_image = image.resize((28, 28))
    image = np.array(resized_image)
    # Invert image (White becomes Black)
    image = 255 - image
    # Mask of image
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # image without background noise removed
    digit_image = cv2.bitwise_and(image, image, mask=mask)
    # Normalize pixel values to range [0, 1]
    image = digit_image / 255.0
    return image


network = load_trained_network()

uploaded_image = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    image_src = convert_to_image_src(uploaded_image)
    cropped_image_src = cropper_component(image_src)

    if cropped_image_src:
        cropped_image = convert_to_image(cropped_image_src)
        preprocessed_image = preprocess_image(cropped_image)
        image = preprocessed_image.reshape(784, 1)
        save_image(image)
        prediction = network.forward_propagation(image)
        save_propability_graph(prediction[:, 0])
