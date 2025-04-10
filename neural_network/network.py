class Network:
    def __init__(self, cost_function):
        self.layers = []
        self.activation_functions = []
        self.cost_function = cost_function

    def add_layer(self, layer, activation_function):
        """
        Fügt eine instanzierte Schicht "layer" mit ihrer entsprechenden
        Aktivierungsfunktion "activation_function" zum Netzwerk hinzu.
        """
        self.layers.append(layer)
        self.activation_functions.append(activation_function)

    def forward_propagation(self, inputs):
        """
        Berechnet die Vorhersagen "predictions" des Netzwerkes anhand der
        Eingabewerte "inputs" der Eingabeschicht.
        """
        current_inputs = inputs
        for layer, activation_function in zip(self.layers, self.activation_functions):
            raw_outputs = layer.forward(current_inputs)
            activated_outputs = activation_function.forward(raw_outputs)
            # Aktivierte Ausgaben der Schicht werden als Eingabewerte
            # für die nächste Schicht verwendet
            current_inputs = activated_outputs
        predictions = current_inputs
        return predictions

    def backpropagation(self, predictions, targets):
        # Gradient der Cost Function in Bezug zu den Vorhersagen (dJ/dY).
        # Beim Categorical Cross Entropy + Softmax sind diese allerdings
        # im Bezug zu den rohen Ausgaben der Ausgabenschicht (dJ/dZ).
        gradient_predictions = self.cost_function.backwards(predictions, targets)
        # Der Gradient der Vorhersagen ist identisch mit dem Gradient der
        # aktivierten Ausgaben der Ausgabenschicht
        gradient_activated_outputs = gradient_predictions
        # Rückwärts berechnet, von Ausgabeschicht zu Eingabeschicht.
        for layer, activation_function in zip(
            reversed(self.layers), reversed(self.activation_functions)
        ):
            # Gradient der Cost Function in Bezug zu den aktivierten Ausgaben
            # der aktuellen Schicht(dJ/dA).
            gradient_raw_outputs = activation_function.backwards(
                gradient_activated_outputs
            )

            # Gradienten der Cost Function in Bezug zu den Gewichten (dJ/dW)
            # und den Bias-Werten der aktuellen Schicht (dJ/db).
            # Berechnet zusätzlich den Gradienten der Cost Function in
            # Bezug zu den rohen Ausgaben der vorherigen Schicht(dJ/dZ).
            gradient_activated_outputs = layer.backwards(gradient_raw_outputs)
