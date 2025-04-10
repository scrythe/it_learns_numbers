from neural_network.load_data import load_trainings_data  # Lade Test und Tranings Daten

from neural_network.network import Network
from neural_network.layer import Layer
from neural_network.activation_functions import ReLU, Softmax
from neural_network.cost_functions import CategoricalCrossEntropy
from neural_network.stochastic_gradient_descent import SGD

import pickle


def create_and_train_network():
    # Categorical Cross Entropy als Cost Function
    network = Network(CategoricalCrossEntropy)
    network.add_layer(
        Layer(784, 20),  # Eingabeschicht → versteckte Schicht
        ReLU(),  # Aktivierungsfunktion für die versteckte Schicht
    )
    network.add_layer(
        Layer(20, 10),  # Versteckte Schicht → Ausgabeschicht
        Softmax(),  # Aktivierungsfunktion für die Ausgabeschicht
    )

    training_images, training_labels = load_trainings_data()

    sgd = SGD(network, learning_rate=0.1)
    sgd.train(
        training_images,
        training_labels,
        batch_size=64,
        desired_avg_cost=0.1,
    )
    return network


def save_network(network):
    with open("network.pickle", "wb") as file:
        pickle.dump(network, file)


network = create_and_train_network()
save_network(network)
