import gzip
import numpy as np


def load_images(file):
    with gzip.open(file, "r") as f:
        f.read(4)  # Überspringen des Headers (Magic Number)
        n_images = int.from_bytes(f.read(4), "big")
        f.read(8)  # Überspringen des Headers (Anzahl Reihen und Zeilen)

        # Lesen der Bilddaten
        # Pixelwerte sind von 0 bis 255 als unsigned Byte gespeichert
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(n_images, 784).T
        images = normalise_images(images)
        return images


def normalise_images(images):
    images = (
        images.reshape(images.shape[0], -1).astype(np.float32) - 127.5
    ) / 127.5  # Zwischen -1 und 1
    return images


def load_labels(file):
    with gzip.open(file, "r") as f:
        f.read(8)  # Überspringen des Headers (Magic Number und Anzahl der Labels)
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        labels = np.eye(10)[labels].T  # Von Ziffern zu "Wahrscheinlichkeiten"
        return labels


def load_trainings_data():
    images = load_images("data/train-images-idx3-ubyte.gz")
    labes = load_labels("data/train-labels-idx1-ubyte.gz")
    return images, labes


def load_test_data():
    images = load_images("data/t10k-images-idx3-ubyte.gz")
    labes = load_labels("data/t10k-labels-idx1-ubyte.gz")
    return images, labes
