import numpy as np  # Import des Pakets NumPy


class SGD:
    def __init__(self, network, learning_rate):
        """
        network: Das Netzwerk, das optimiert werden soll
        learning_rate: Die Lernrate, die die Schrittgröße bestimmt
        """
        self.network = network
        self.learning_rate = learning_rate

    def update_parameters(self):
        """
        Aktualisiert die Parameter (Gewichte und Bias-Werte) aller
        Schichten im Netzwerk basierend auf den Gradienten
        """
        # Iteriert über alle Schichten des Netzwerks und aktualisiert deren
        # Parameter
        for layer in self.network.layers:
            # Aktualisiert die Gewichte der aktuellen Schicht mit dem
            # negativen Gradienten multipliziert mit der Lernrate, um den
            # Schritt zu skalieren
            layer.weights -= self.learning_rate * layer.gradient_weights
            # Aktualisiert die Bias-Werte der aktuellen Schicht mit dem
            # negativen Gradienten multipliziert mit der Lernrate, um
            # sden Schritt zu skalieren
            layer.biases -= self.learning_rate * layer.gradient_biases

    def create_batches(self, inputs, targets, batch_size):
        N = inputs.shape[1]  # Anzahl an Trainingsdaten
        # Trainingsset in mini Batches eingeteilt
        batches = []
        for i in range(0, N, batch_size):
            batch_inputs = inputs[:, i : i + batch_size]
            batch_targets = targets[:, i : i + batch_size]
            single_batch = (batch_inputs, batch_targets)
            batches.append(single_batch)
        return batches

    def optimise(
        self,
        inputs,
        targets,
    ):
        # Vorwärtsdurchlauf: Berechnung der Vorhersagen
        predictions = self.network.forward_propagation(inputs)

        cost = self.network.cost_function.calculate_cost(predictions, targets)

        # Rückwärtsdurchlauf: Berechnung der Gradienten
        self.network.backpropagation(predictions, targets)

        # Aktualisiert die Gewichte und Bias-Werte basierend auf
        # die Gradienten
        self.update_parameters()
        return cost

    def train(self, inputs, targets, batch_size, desired_avg_cost):
        batches = self.create_batches(inputs, targets, batch_size)
        cost_history = []
        avg_epoch_cost = 1
        while avg_epoch_cost > desired_avg_cost:
            epoch_cost_history = []
            for batch_inputs, batch_targets in batches:
                cost = self.optimise(batch_inputs, batch_targets)
                self.optimise(batch_inputs, batch_targets)

                epoch_cost_history.append(cost)
                cost_history.append(cost)
            avg_epoch_cost = np.mean(epoch_cost_history)
            print(avg_epoch_cost)

        return cost_history
