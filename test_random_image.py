import numpy as np  # Import des Pakets NumPy
import pickle
import random
import matplotlib.pyplot as plt
from PIL import Image

from neural_network.load_data import load_test_data  # Lade Test und Tranings Daten


def get_random_image_dataset():
    images, _ = load_test_data()
    n_images = images.shape[1]
    rand_i = random.randrange(0, n_images)
    image = images[:, rand_i]
    image = image.reshape(len(image), 1)
    return image


def save_image(image):
    plt.imshow(image.reshape(28, 28), cmap="Greys")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("image.png")
    plt.close()


def save_propability_graph(prediction):
    digits = np.arange(10)
    plt.bar(digits, prediction)

    for i, prob in enumerate(prediction):
        plt.text(i, prob * 1.2, f"{prob:.1e}", ha="center", va="bottom", fontsize=10)

    plt.xticks(digits)
    plt.ylim(1e-40, 1e1)
    plt.xlabel("Ziffer")
    plt.ylabel("Wahrscheinlichkeit")
    plt.yscale("log")  # Log scale to better visualize small values
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("probability.png")
    plt.close()


def load_trained_network():
    with open("network.pickle", "rb") as file:
        network = pickle.load(file)
    return network


network = load_trained_network()
image = get_random_image_dataset()
prediction = network.forward_propagation(image)[:, 0]
save_image(image)
save_propability_graph(prediction)
