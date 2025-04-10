import numpy as np  # Import des Pakets NumPy


class Sigmoid:
    def forward(self, raw_outputs):
        """
        Berechnet die aktivierten Ausgabewerte basierend auf den rohen
        Ausgabewerten "raw_outputs".
        """
        activated_outputs = 1 / (1 + np.exp(-raw_outputs))
        # Aktivierte Ausgaben für spätere Verwendung speichern
        self.saved_activated_outputs = activated_outputs
        return activated_outputs

    def backwards(self, gradient_activated_outputs):
        """
        Berechnet den Gradienten der Cost Function in Bezug zu
        den rohen Ausgaben der aktuellen Schicht (dJ/dZ)

        gradient_activated_outputs: Gradient der Cost Function in Bezug
        zu den aktivierten Ausgaben der aktuellen Schicht (dJ/dA)
        """
        # Gradient in Bezug zu (A*(A-1))
        d_activated_d_raw = self.saved_activated_outputs * (
            1 - self.saved_activated_outputs
        )

        gradient_raw_outputs = gradient_activated_outputs * d_activated_d_raw
        return gradient_raw_outputs


class ReLU:
    def forward(self, raw_outputs):
        """
        Berechnet die aktivierten Ausgabewerte basierend auf den rohen
        Ausgabewerten "raw_outputs".
        """
        self.saved_raw_outputs = raw_outputs
        activated_outputs = np.maximum(0, raw_outputs)
        return activated_outputs

    def backwards(self, gradient_activated_outputs):
        """
        Berechnet den Gradienten der Cost Function in Bezug zu
        den rohen Ausgaben der aktuellen Schicht (dJ/dZ).

        gradient_activated_outputs: Gradient der Cost Function in Bezug
        zu den aktivierten Ausgaben der aktuellen Schicht (dJ/dA).
        """
        # Gradient der Cost Function in Bezug zu den rohen Ausgaben (dJ/dZ).
        gradient_raw_outputs = gradient_activated_outputs * (self.saved_raw_outputs > 0)
        return gradient_raw_outputs


class Softmax:
    def forward(self, raw_outputs):
        """
        Berechnet die aktivierten Ausgabewerte basierend auf den rohen
        Ausgabewerten "raw_outputs".
        """
        # Exponierte Werte
        exponentiated_values = np.exp(raw_outputs - np.max(raw_outputs, axis=0))
        # Summe der exponierten Werte
        sum_values = np.sum(exponentiated_values, axis=0, keepdims=True)
        # Normalisierte / aktivierte Ausgaben
        normalized_outputs = exponentiated_values / sum_values
        return normalized_outputs

    def backwards(self, gradient_raw_outputs):
        # Gibt die Gradienten direkt weiter (Softmax wird in Kombination
        # mit Categorical Cross Entropy verwendet).
        return gradient_raw_outputs
