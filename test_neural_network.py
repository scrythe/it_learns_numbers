import numpy as np  # Import des Pakets NumPy
import pickle

from neural_network.load_data import load_test_data  # Lade Test und Tranings Daten


def test_neural_network(network):
    # Bilder (Eingabewerte) und labels (tatsächliche Zielwerte als
    # Wahrscheinlichkeiten)
    images, labels = load_test_data()

    # Vorhersagen als Wahrscheinlichkeitsverteilung
    predictions = network.forward_propagation(images)

    N = predictions.shape[1]  # Anzahl an Trainingsbeispielen

    # Vorhersagen als Ziffern
    predicted_numbers = np.argmax(predictions, axis=0)

    # tatsächliche Zielwerte als Ziffern
    actual_values = np.argmax(labels, axis=0)

    # Vektor aus "Richtig Falsch" Werten
    comparisons = predicted_numbers == actual_values

    # Summe / Anzahl an richtigen Aussagen
    n_correct_predictions = sum(comparisons)

    # Genauigkeit des neuronalen Netzwerkes
    accuracy = n_correct_predictions / N

    print(accuracy)


def load_trained_network():
    with open("network.pickle", "rb") as file:
        network = pickle.load(file)
    return network


network = load_trained_network()
test_neural_network(network)
