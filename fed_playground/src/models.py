"""Machine learning model implementations for federated learning.

All models implement the :class:`Model` ABC, which defines the five-method
interface expected by the rest of the framework: :meth:`~Model.train`,
:meth:`~Model.get_parameters`, :meth:`~Model.set_parameters`,
:meth:`~Model.predict`, and :meth:`~Model.evaluate`.
"""

import abc

import numpy as np


class Model(abc.ABC):
    """Abstract base class for machine learning models.

    Subclasses must implement all five abstract methods so that they can be
    used interchangeably by :class:`~fed_playground.src.party.Party` and
    :class:`~fed_playground.src.environment.Environment`.
    """

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the provided data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """

    @abc.abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Return the current model parameters as a flat numpy array.

        Returns:
            1-D numpy array containing all trainable parameters.
        """

    @abc.abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        """Replace the current model parameters.

        Args:
            params: Flat numpy array of the same length as returned by
                :meth:`get_parameters`.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions for the given feature matrix.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """

    @abc.abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute a scalar evaluation metric (MSE by default).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True target values of shape ``(n_samples,)``.

        Returns:
            Mean Squared Error on the provided data.

        Raises:
            ValueError: If *X* or *y* contain zero samples.
        """


class LinearRegressionModel(Model):
    """Linear regression via mini-batch gradient descent.

    Parameters are stored as a concatenation of ``[weights, bias]`` so that
    they can be serialised and exchanged over the federated network as a single
    flat array.

    Args:
        input_dim: Number of input features.
        learning_rate: Step size for gradient descent updates (default ``0.01``).
        epochs: Number of full passes over the training data per :meth:`train`
            call (default ``1`` for federated settings where data is small).
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.01,
        epochs: int = 1,
    ) -> None:
        self.input_dim = input_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train for ``self.epochs`` passes using MSE gradient descent.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            return

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            error = y - y_pred
            dw = -(2.0 / n_samples) * np.dot(X.T, error)
            db = -(2.0 / n_samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def get_parameters(self) -> np.ndarray:
        """Return ``[weights..., bias]`` as a flat array.

        Returns:
            1-D numpy array of length ``input_dim + 1``.
        """
        return np.concatenate([self.weights, [self.bias]])

    def set_parameters(self, params: np.ndarray) -> None:
        """Load weights and bias from a flat parameter array.

        Args:
            params: 1-D array of length ``input_dim + 1``; last element is bias.
        """
        self.weights = params[:-1]
        self.bias = float(params[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute linear predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values ``X @ weights + bias``.
        """
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error on the provided data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True target values of shape ``(n_samples,)``.

        Returns:
            MSE scalar.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))


class RidgeRegressionModel(Model):
    """L2-regularised linear regression via the closed-form ridge solution.

    Solves ``w = (XᵀX + αI)⁻¹Xᵀy`` exactly.  The bias term is excluded from
    regularisation, following standard practice.

    Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        alpha: L2 regularisation strength (default ``1.0``).
    """

    def __init__(self, input_dim: int, alpha: float = 1.0) -> None:
        self.input_dim = input_dim
        self.alpha = alpha
        self.params: np.ndarray = np.zeros(input_dim + 1)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the ridge model using the closed-form solution.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        if X.shape[0] == 0:
            return
        n = X.shape[0]
        X_b = np.hstack([X, np.ones((n, 1))])
        reg = self.alpha * np.eye(X_b.shape[1])
        reg[-1, -1] = 0.0  # do not regularise the bias term
        self.params = np.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)

    def get_parameters(self) -> np.ndarray:
        """Return ``[weights..., bias]`` as a flat array."""
        return self.params

    def set_parameters(self, params: np.ndarray) -> None:
        """Load parameters from a flat array.

        Args:
            params: 1-D array of length ``input_dim + 1``.
        """
        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute linear predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_b @ self.params

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE on the provided data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True target values of shape ``(n_samples,)``.

        Returns:
            MSE scalar.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        return float(np.mean((y - self.predict(X)) ** 2))


class LogisticRegressionModel(Model):
    """Binary logistic regression via gradient descent.

    Uses the sigmoid activation and binary cross-entropy loss.  Targets
    *y* must be in ``{0, 1}``.  The :meth:`evaluate` method returns
    **accuracy** (fraction of correct predictions) rather than MSE so
    that downstream comparison with regression models is intentionally
    visible.

    Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        learning_rate: Gradient descent step size (default ``0.1``).
        epochs: Passes over training data per :meth:`train` call (default ``10``).
        threshold: Decision boundary for binary prediction (default ``0.5``).
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.1,
        epochs: int = 10,
        threshold: float = 0.5,
    ) -> None:
        self.input_dim = input_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train for ``self.epochs`` passes using sigmoid cross-entropy gradients.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary target vector of shape ``(n_samples,)`` with values in {0, 1}.
        """
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(self.epochs):
            prob = self._sigmoid(X @ self.weights + self.bias)
            error = prob - y
            self.weights -= self.lr * (X.T @ error) / n
            self.bias -= self.lr * float(np.sum(error)) / n

    def get_parameters(self) -> np.ndarray:
        """Return ``[weights..., bias]`` as a flat array."""
        return np.concatenate([self.weights, [self.bias]])

    def set_parameters(self, params: np.ndarray) -> None:
        """Load weights and bias from a flat parameter array.

        Args:
            params: 1-D array of length ``input_dim + 1``.
        """
        self.weights = params[:-1].copy()
        self.bias = float(params[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard binary predictions (0 or 1) based on ``self.threshold``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Binary predictions of shape ``(n_samples,)``.
        """
        prob = self._sigmoid(X @ self.weights + self.bias)
        return (prob >= self.threshold).astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities P(y=1 | x).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probabilities of shape ``(n_samples,)`` in ``[0, 1]``.
        """
        return self._sigmoid(X @ self.weights + self.bias)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True binary labels of shape ``(n_samples,)``.

        Returns:
            Accuracy in ``[0, 1]`` (fraction of correct predictions).

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        return float(np.mean(self.predict(X) == y))


class MLPRegressorModel(Model):
    """Single-hidden-layer MLP regressor implemented in pure NumPy.

    Architecture: ``input_dim → hidden_dim → 1`` with ReLU activation on
    the hidden layer and a linear output neuron.  Trained via full-batch
    gradient descent and MSE loss.

    Parameters are flattened as
    ``[W1 (input_dim × hidden_dim), b1 (hidden_dim,), W2 (hidden_dim,), b2 (1,)]``.

    Args:
        input_dim: Number of input features.
        hidden_dim: Width of the hidden layer (default ``16``).
        learning_rate: Gradient descent step size (default ``0.01``).
        epochs: Passes over training data per :meth:`train` call (default ``5``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 5,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs
        rng = np.random.default_rng(0)
        scale = np.sqrt(2.0 / input_dim)
        self.W1: np.ndarray = rng.normal(0, scale, (input_dim, hidden_dim))
        self.b1: np.ndarray = np.zeros(hidden_dim)
        self.W2: np.ndarray = rng.normal(0, 0.01, hidden_dim)
        self.b2: float = 0.0

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_pre = X @ self.W1 + self.b1  # (n, hidden_dim)
        h = np.maximum(0.0, h_pre)      # ReLU
        out = h @ self.W2 + self.b2     # (n,)
        return h_pre, h, out

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train for ``self.epochs`` passes using MSE backpropagation.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(self.epochs):
            h_pre, h, out = self._forward(X)
            # MSE loss: L = mean((out - y)^2)
            delta_out = (out - y) / n          # (n,)
            dW2 = h.T @ delta_out              # (hidden_dim,)
            db2 = float(np.sum(delta_out))
            delta_h = np.outer(delta_out, self.W2) * (h_pre > 0)  # (n, hidden_dim)
            dW1 = X.T @ delta_h                # (input_dim, hidden_dim)
            db1 = delta_h.sum(axis=0)          # (hidden_dim,)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def get_parameters(self) -> np.ndarray:
        """Return all parameters as a single flat array.

        Returns:
            1-D numpy array of length
            ``input_dim × hidden_dim + hidden_dim + hidden_dim + 1``.
        """
        return np.concatenate(
            [self.W1.ravel(), self.b1, self.W2, [self.b2]]
        )

    def set_parameters(self, params: np.ndarray) -> None:
        """Load all parameters from a flat array.

        Args:
            params: 1-D array produced by :meth:`get_parameters`.
        """
        d, h = self.input_dim, self.hidden_dim
        idx = 0
        self.W1 = params[idx : idx + d * h].reshape(d, h)
        idx += d * h
        self.b1 = params[idx : idx + h]
        idx += h
        self.W2 = params[idx : idx + h]
        idx += h
        self.b2 = float(params[idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute regression predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """
        _, _, out = self._forward(X)
        return out

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE on the provided data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True target values of shape ``(n_samples,)``.

        Returns:
            MSE scalar.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        return float(np.mean((y - self.predict(X)) ** 2))


class ClosedFormLinearRegressionModel(Model):
    """Linear regression via the normal equation (closed-form solution).

    Solves ``w = argmin ||Xw - y||²`` using :func:`numpy.linalg.lstsq`, which
    is numerically stable even for rank-deficient design matrices.

    Parameters are stored as a single flat array ``[weights..., bias]`` where
    the bias corresponds to an implicit column of ones appended to *X*.

    Args:
        input_dim: Number of input features.
    """

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        # weights + 1 bias term
        self.params: np.ndarray = np.zeros(input_dim + 1)

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Append a column of ones to *X* for the bias term.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Augmented matrix of shape ``(n_samples, n_features + 1)``.
        """
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Solve the normal equations via least-squares.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        if X.shape[0] == 0:
            return
        X_b = self._add_bias_term(X)
        # rcond=None uses machine precision as the cutoff for small singular values.
        self.params, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

    def get_parameters(self) -> np.ndarray:
        """Return ``[weights..., bias]`` as a flat array.

        Returns:
            1-D numpy array of length ``input_dim + 1``.
        """
        return self.params

    def set_parameters(self, params: np.ndarray) -> None:
        """Load parameters from a flat array.

        Args:
            params: 1-D array of length ``input_dim + 1``.
        """
        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute linear predictions using the bias-augmented design matrix.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """
        return np.dot(self._add_bias_term(X), self.params)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error on the provided data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True target values of shape ``(n_samples,)``.

        Returns:
            MSE scalar.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))
