from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import tensorflow as tf


class MNISTClassifier(abc.ABC):
    """Base class for MNIST digit classifiers."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return a compiled Keras model."""

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 128,
              validation_split: float = 0.1) -> tf.keras.callbacks.History:
        """Train the model on the given data.

        TODO: Implement this method.
        - If self.model is None, call self.build_model() to create it.
        - Use the model's fit() method with the provided parameters.
        - Return the History object from fit().
        """
        raise NotImplementedError

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's evaluate() method to get loss and accuracy.
        - Use the model's predict() method and np.argmax to get predicted labels.
        - Return a dict with keys: "loss", "accuracy", "y_pred".
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the model to the given file path.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's save() method.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load a model from the given file path.

        TODO: Implement this method.
        - Use tf.keras.models.load_model() and assign to self.model.
        """
        raise NotImplementedError


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        """Build a logistic regression model for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - A single Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="sgd", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        """Build a simple neural network for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - Dense hidden layer with 128 units and ReLU activation.
          - Dense hidden layer with 64 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Autograde self-test — DO NOT MODIFY ANYTHING BELOW THIS LINE
#
# The autograder runs:  python classifier.py
# It builds both models and checks your implementation on random data
# (no dataset needed). Exit code 0 = pass. Run it yourself before submitting.
# ──────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import os

    rng = np.random.default_rng(0)
    x = rng.random((8, 784), dtype=np.float32)
    y = rng.integers(0, 10, size=(8,))

    checks = [
        ("logistic", LogisticRegressionClassifier, 7_850),
        ("nn", NeuralNetworkClassifier, 109_386),
    ]

    for name, cls, expected_params in checks:
        clf = cls()
        model = clf.build_model()
        assert model is not None, f"{name}: build_model() returned None"
        n_params = model.count_params()
        assert n_params == expected_params, (
            f"{name}: expected {expected_params:,} parameters, got {n_params:,}"
            " — check your layer sizes and activations"
        )
        probs = model.predict(x, verbose=0)
        assert probs.shape == (8, 10), (
            f"{name}: output shape is {probs.shape}, expected (8, 10)"
        )
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), (
            f"{name}: outputs do not sum to 1 — did you use softmax?"
        )
        clf.model = model
        history = clf.train(x, y, epochs=1, batch_size=4)
        assert history is not None, f"{name}: train() must return the History object"
        result = clf.evaluate(x, y)
        for key in ("loss", "accuracy", "y_pred"):
            assert key in result, f"{name}: evaluate() result is missing key '{key}'"
        assert len(np.asarray(result["y_pred"])) == 8, (
            f"{name}: y_pred must contain one predicted label per sample"
        )
        path = f"_selftest_{name}.keras"
        clf.save(path)
        assert os.path.exists(path), f"{name}: save() did not create {path}"
        clf.load(path)
        assert clf.model is not None, f"{name}: load() did not set self.model"
        os.remove(path)
        print(f"[PASS] {name}")

    print("Self-test passed. Your classifier.py is ready to submit.")


if __name__ == "__main__":
    _self_test()
