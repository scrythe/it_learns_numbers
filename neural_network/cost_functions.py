import numpy as np  # Import des Pakets NumPy


class MeanSquaredError:
    def calculate_cost(predictions, targets):
        losses = np.sum(np.square(predictions - targets), axis=0)
        cost = np.mean(losses)
        return cost

    def backwards(predictions, targets):
        """
        Berechnet den Gradienten des Mean Squared Error in Bezug zu den
        Vorhersagen.
        """
        N = predictions.shape[1]  # Anzahl an Trainingsbeispielen
        gradient_predictions = (2 / N) * (predictions - targets) / len(predictions)
        return gradient_predictions


class CategoricalCrossEntropy:
    def calculate_cost(predictions, targets):
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        losses = -np.sum(targets * np.log(predictions), axis=0)
        cost = np.mean(losses)
        return cost

    def backwards(predictions, targets):
        """
        Berechnet den Gradienten des Categorical Cross Entropy in
        Bezug zu den rohen Ausgaben der Ausgabenschicht (dJ/dZ).
        """
        N = predictions.shape[1]  # Anzahl an Trainingsbeispielen
        gradient_raw_outputs = (predictions - targets) / N
        return gradient_raw_outputs
