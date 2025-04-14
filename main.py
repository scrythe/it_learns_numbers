import streamlit as st
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from streamlit_cropper import st_cropper


def load_trained_network():
    with open("network.pickle", "rb") as file:
        network = pickle.load(file)
    return network


def save_image(image):
    # image = image.reshape(28, 28)
    fig, _ = plt.subplots()
    plt.imshow(image, cmap="Greys")
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


def roi(image):
    # Mask of image
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image to the bounding rectangle
        cropped = image[y : y + h, x : x + w]
        return cropped


def preprocess_image(image):
    # Mask of image
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Image background noise removed
    digit_image = cv2.bitwise_and(image, image, mask=mask)
    # Normalize pixel values to range [0, 1]
    image = digit_image / 255.0
    return image


def add_padding(image, padding):
    padded_image = np.zeros((28, 28))

    offset = padding // 2

    # Paste the 20x20 digit into the center of the 28x28 canvas
    padded_image[offset : 28 - offset, offset : 28 - offset] = image

    return padded_image


network = load_trained_network()

uploaded_image = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image)
    cropped_image = st_cropper(image, aspect_ratio=(1, 1)).convert("L")
    image_data = np.array(cropped_image)
    # Invert image (White becomes Black)
    image_data = 255 - image_data
    cropped_image = roi(image_data)
    padding = 12
    resized = cv2.resize(
        cropped_image, (28 - padding, 28 - padding), interpolation=cv2.INTER_AREA
    )
    preprocessed_image = preprocess_image(resized)
    padded_image = add_padding(preprocessed_image, padding)

    save_image(padded_image)
    image_data = padded_image.reshape(784, 1)
    prediction = network.forward_propagation(image_data)
    save_propability_graph(prediction[:, 0])
