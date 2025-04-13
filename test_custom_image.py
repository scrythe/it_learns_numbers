import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle


def normalise_image(image):
    image = (
        127.5 - image.reshape(image.shape[0], -1).astype(np.float32)
    ) / 127.5  # Zwischen -1 und 1
    return image


def load_trained_network():
    with open("network.pickle", "rb") as file:
        network = pickle.load(file)
    return network


def load_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(784, 1)
    image = image / 255  # Zwischen -1 und 1
    image = 1 - image
    return image


def save_image(image):
    plt.imshow(image.reshape(28, 28), cmap="Greys")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("converted_image.png")
    plt.close()


def save_propability_graph(prediction):
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
    plt.savefig("probability.png")
    plt.close()


network = load_trained_network()
image = load_image("image.png")
save_image(image)
prediction = network.forward_propagation(image)
save_propability_graph(prediction[:, 0])
