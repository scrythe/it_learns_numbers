import numpy as np  # Import des Pakets NumPy


class Layer:
    def __init__(self, n_inputs, n_neurons):
        """
        n_inputs: Anzahl an Eingabewerten (bzw. Neuronen der vorherigen
        Schicht).
        n_neurons: Anzahl an Neuronen für diese Schicht.
        """
        # Gewichtsmatrix
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        # Bias-Vektor
        self.biases = 0.1 * np.random.randn(n_neurons, 1)

    def forward(self, inputs):
        """
        Berechnung des Ausgabewerts für die Neuronen in dieser Schicht
        basierend auf den Eingabewerte "inputs".
        """
        # Eingabewerte für spätere Verwendung speichern
        self.saved_inputs = inputs
        # Ausgabewerte als Matrix
        outputs = np.dot(self.weights, inputs) + self.biases
        return outputs  # Rückgabe der Ausgabewerte

    def backwards(self, gradient_raw_outputs):
        """
        Berechnet den Gradienten der Cost Function in Bezug zu den
        Gewichten und Bias-Werten der aktuellen Schicht und aktivierten
        Ausgaben der vorherigen Schicht.

        gradient_raw_outputs: Gradient der Cost Function in Bezug zu den
        rohen Ausgaben der aktuellen Schicht (dJ/dZ).
        """

        # Gradient der Cost Function in Bezug zu den Gewichten von der
        # aktuellen Schicht (dJ/dW).
        self.gradient_weights = np.dot(
            gradient_raw_outputs,
            self.saved_inputs.T,
        )

        # Gradient in Bezug zu den Bias-Werten (dJ/db).
        self.gradient_biases = np.sum(gradient_raw_outputs, axis=1, keepdims=True)

        # Gradient in Bezug zu den aktivierten Ausgaben der vorherigen
        # Schicht (dJ/dA).
        gradient_activated_outputs = np.dot(self.weights.T, gradient_raw_outputs)
        return gradient_activated_outputs
