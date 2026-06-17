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

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error on the provided data.

        This is the default metric for regression models.  Classification
        models override it to return accuracy instead.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True target values of shape ``(n_samples,)``.

        Returns:
            Mean Squared Error on the provided data.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))


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

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_pre = X @ self.W1 + self.b1  # (n, hidden_dim)
        h = np.maximum(0.0, h_pre)  # ReLU
        out = h @ self.W2 + self.b2  # (n,)
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
            delta_out = (out - y) / n  # (n,)
            dW2 = h.T @ delta_out  # (hidden_dim,)
            db2 = float(np.sum(delta_out))
            delta_h = np.outer(delta_out, self.W2) * (h_pre > 0)  # (n, hidden_dim)
            dW1 = X.T @ delta_h  # (input_dim, hidden_dim)
            db1 = delta_h.sum(axis=0)  # (hidden_dim,)
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
        return np.concatenate([self.W1.ravel(), self.b1, self.W2, [self.b2]])

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


class MLPClassifierModel(Model):
    """Single-hidden-layer MLP classifier with softmax output (pure NumPy).

    Architecture: ``input_dim → hidden_dim → n_classes`` with ReLU hidden
    activation and softmax output.  Trained via mini-batch SGD with
    cross-entropy loss.

    :meth:`evaluate` returns **accuracy** (fraction correct) rather than
    MSE, making it suitable for classification tasks such as MNIST.

    Parameters are flattened as
    ``[W1 (input_dim × hidden_dim), b1 (hidden_dim,),
       W2 (hidden_dim × n_classes), b2 (n_classes,)]``.

    Args:
        input_dim: Number of input features.
        hidden_dim: Width of the hidden layer (default ``64``).
        n_classes: Number of output classes (default ``10``).
        learning_rate: SGD step size (default ``0.01``).
        epochs: Passes over the local data per :meth:`train` call (default ``3``).
        batch_size: Mini-batch size (default ``64``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_classes: int = 10,
        learning_rate: float = 0.01,
        epochs: int = 3,
        batch_size: int = 64,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        rng = np.random.default_rng(0)
        self.W1: np.ndarray = rng.normal(
            0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)
        )
        self.b1: np.ndarray = np.zeros(hidden_dim)
        self.W2: np.ndarray = rng.normal(
            0, np.sqrt(2.0 / hidden_dim), (hidden_dim, n_classes)
        )
        self.b2: np.ndarray = np.zeros(n_classes)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_stable = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z_stable)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_pre = X @ self.W1 + self.b1
        h = np.maximum(0.0, h_pre)
        probs = self._softmax(h @ self.W2 + self.b2)
        return h_pre, h, probs

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train with mini-batch SGD and cross-entropy loss.

        Args:
            X: Feature matrix of shape ``(n_samples, input_dim)``.
            y: Integer class labels of shape ``(n_samples,)``
                with values in ``{0, …, n_classes - 1}``.
        """
        n = X.shape[0]
        if n == 0:
            return
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y.astype(int)] = 1.0

        rng = np.random.default_rng()
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                Xb, Yb = X[idx], Y[idx]
                nb = Xb.shape[0]
                h_pre, h, probs = self._forward(Xb)
                d_out = (probs - Yb) / nb  # (nb, n_classes)
                dW2 = h.T @ d_out  # (hidden_dim, n_classes)
                db2 = d_out.sum(axis=0)
                d_h = (d_out @ self.W2.T) * (h_pre > 0)  # ReLU mask
                dW1 = Xb.T @ d_h
                db1 = d_h.sum(axis=0)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

    def get_parameters(self) -> np.ndarray:
        """Return all parameters as a single flat array."""
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_parameters(self, params: np.ndarray) -> None:
        """Load all parameters from a flat array.

        Args:
            params: 1-D array produced by :meth:`get_parameters`.
        """
        d, h, c = self.input_dim, self.hidden_dim, self.n_classes
        idx = 0
        self.W1 = params[idx : idx + d * h].reshape(d, h).copy()
        idx += d * h
        self.b1 = params[idx : idx + h].copy()
        idx += h
        self.W2 = params[idx : idx + h * c].reshape(h, c).copy()
        idx += h * c
        self.b2 = params[idx : idx + c].copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class index for each sample.

        Args:
            X: Feature matrix of shape ``(n_samples, input_dim)``.

        Returns:
            Integer class predictions of shape ``(n_samples,)``.
        """
        _, _, probs = self._forward(X)
        return probs.argmax(axis=1).astype(float)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix of shape ``(n_samples, input_dim)``.
            y: True integer class labels of shape ``(n_samples,)``.

        Returns:
            Accuracy in ``[0, 1]``.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        return float(np.mean(self.predict(X) == y.astype(int)))


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


class SVMModel(Model):
    """Linear support vector machine trained with the Pegasos algorithm.

    Reference: Shalev-Shwartz, Singer, Srebro & Cotter, "Pegasos: Primal
    Estimated sub-GrAdient SOlver for SVM", Mathematical Programming 2011.

    Minimises the L2-regularised hinge loss
    ``(λ/2)‖w‖² + (1/n) Σ max(0, 1 - yᵢ(w·xᵢ + b))`` by stochastic sub-gradient
    descent with the Pegasos step size ``η_t = 1/(λt)``.  Binary labels are
    supplied in ``{0, 1}`` and mapped internally to ``{-1, +1}``; the bias term
    is left unregularised.  :meth:`evaluate` returns **accuracy**.

    Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        lambda_reg: Regularisation strength ``λ`` (default ``0.01``).
        epochs: Passes over the local data per :meth:`train` call (default ``20``).
        seed: RNG seed for the sample order (default ``0``).
    """

    def __init__(
        self,
        input_dim: int,
        lambda_reg: float = 0.01,
        epochs: int = 20,
        seed: int = 0,
    ) -> None:
        if lambda_reg <= 0:
            raise ValueError("lambda_reg must be > 0.")
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.seed = seed
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0
        self._t = 0  # global Pegasos step counter (persists across rounds)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run Pegasos sub-gradient updates over the local data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary labels of shape ``(n_samples,)`` with values in {0, 1}.
        """
        n = X.shape[0]
        if n == 0:
            return
        y_pm = np.where(y > 0.5, 1.0, -1.0)  # {0,1} -> {-1,+1}
        rng = np.random.default_rng(self.seed + self._t)
        for _ in range(self.epochs):
            for i in rng.permutation(n):
                self._t += 1
                eta = 1.0 / (self.lambda_reg * self._t)
                margin = y_pm[i] * (self.weights @ X[i] + self.bias)
                self.weights *= 1.0 - eta * self.lambda_reg
                if margin < 1.0:
                    self.weights += eta * y_pm[i] * X[i]
                    self.bias += eta * y_pm[i]

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

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the signed distance ``w·x + b`` to the separating hyperplane.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Real-valued scores of shape ``(n_samples,)``.
        """
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard binary predictions in ``{0, 1}``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Binary predictions of shape ``(n_samples,)``.
        """
        return (self.decision_function(X) >= 0.0).astype(float)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True binary labels of shape ``(n_samples,)``.

        Returns:
            Accuracy in ``[0, 1]``.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        return float(np.mean(self.predict(X) == y))


class LassoRegressionModel(Model):
    """L1-regularised linear regression (LASSO) via cyclic coordinate descent.

    References: Tibshirani, "Regression Shrinkage and Selection via the Lasso",
    JRSS-B 1996; Friedman, Hastie & Tibshirani, "Regularization Paths for
    Generalized Linear Models via Coordinate Descent", J. Stat. Soft. 2010.

    Minimises ``(1/2n)‖y - Xw - b‖² + α‖w‖₁`` with cyclic coordinate descent and
    soft-thresholding ``S(ρ, α) = sign(ρ)·max(|ρ| - α, 0)``.  The L1 penalty
    drives weights of irrelevant features to *exactly* zero, yielding a sparse,
    feature-selecting model.  The intercept is not penalised.

    Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        alpha: L1 regularisation strength (default ``0.1``).
        max_iter: Maximum coordinate-descent sweeps per :meth:`train` (default ``1000``).
        tol: Convergence tolerance on the max coefficient change (default ``1e-6``).
    """

    def __init__(
        self,
        input_dim: int,
        alpha: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.input_dim = input_dim
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0

    @staticmethod
    def _soft_threshold(rho: float, gamma: float) -> float:
        """Soft-thresholding operator ``S(rho, gamma)``."""
        if rho > gamma:
            return rho - gamma
        if rho < -gamma:
            return rho + gamma
        return 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit by cyclic coordinate descent with soft-thresholding.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        n = X.shape[0]
        if n == 0:
            return
        # Per-feature normalisation z_j = (1/n) Σ x_ij²; constant features stay 0.
        col_sq = (X * X).sum(axis=0) / n
        for _ in range(self.max_iter):
            max_change = 0.0
            self.bias = float(np.mean(y - X @ self.weights))
            for j in range(self.input_dim):
                if col_sq[j] == 0.0:
                    continue
                # Partial residual excluding feature j's current contribution.
                r_j = y - self.bias - X @ self.weights + self.weights[j] * X[:, j]
                rho_j = float(X[:, j] @ r_j) / n
                new_wj = self._soft_threshold(rho_j, self.alpha) / col_sq[j]
                max_change = max(max_change, abs(new_wj - self.weights[j]))
                self.weights[j] = new_wj
            if max_change < self.tol:
                break

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
        """Compute linear predictions ``X·weights + bias``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """
        return X @ self.weights + self.bias


class ElasticNetRegressionModel(Model):
    """Elastic-net regression: blended L1/L2 penalty via coordinate descent.

    Reference: Zou & Hastie, "Regularization and variable selection via the
    elastic net", JRSS-B 2005.

    Minimises ``(1/2n)‖y - Xw - b‖² + α·ρ‖w‖₁ + (α(1-ρ)/2)‖w‖²`` where ``ρ`` is
    the L1 ratio.  The L1 part yields sparsity; the L2 part stabilises selection
    among correlated features (the "grouping effect") that pure LASSO handles
    poorly.  Coordinate update with soft-thresholding::

        w_j = S((1/n) xⱼ·r_j, α·ρ) / ((1/n)‖xⱼ‖² + α(1-ρ))

    Recovers LASSO at ``l1_ratio=1`` and ridge-like shrinkage at ``l1_ratio=0``.
    The intercept is unpenalised.  Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        alpha: Overall regularisation strength (default ``0.1``).
        l1_ratio: Mix between L1 (``1.0``) and L2 (``0.0``) penalties (default ``0.5``).
        max_iter: Maximum coordinate-descent sweeps (default ``1000``).
        tol: Convergence tolerance on the max coefficient change (default ``1e-6``).
    """

    def __init__(
        self,
        input_dim: int,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1].")
        self.input_dim = input_dim
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0

    @staticmethod
    def _soft_threshold(rho: float, gamma: float) -> float:
        if rho > gamma:
            return rho - gamma
        if rho < -gamma:
            return rho + gamma
        return 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit by cyclic coordinate descent with an L1/L2-blended penalty.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        n = X.shape[0]
        if n == 0:
            return
        col_sq = (X * X).sum(axis=0) / n
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        for _ in range(self.max_iter):
            max_change = 0.0
            self.bias = float(np.mean(y - X @ self.weights))
            for j in range(self.input_dim):
                denom = col_sq[j] + l2
                if denom == 0.0:
                    continue
                r_j = y - self.bias - X @ self.weights + self.weights[j] * X[:, j]
                rho_j = float(X[:, j] @ r_j) / n
                new_wj = self._soft_threshold(rho_j, l1) / denom
                max_change = max(max_change, abs(new_wj - self.weights[j]))
                self.weights[j] = new_wj
            if max_change < self.tol:
                break

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
        """Compute linear predictions ``X·weights + bias``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """
        return X @ self.weights + self.bias


class PoissonRegressionModel(Model):
    """Poisson regression (log-linear GLM) for non-negative count targets.

    Reference: Nelder & Wedderburn, "Generalized Linear Models", JRSS-A 1972.

    Models ``E[y | x] = exp(w·x + b)`` (the canonical log link for the Poisson
    family) and fits by gradient descent on the negative log-likelihood, whose
    gradient is the elegant ``(1/n) Xᵀ(μ - y)`` with ``μ = exp(w·x + b)``.
    Suitable for rates/counts (events per unit), where ordinary least squares
    would wrongly allow negative predictions.

    :meth:`predict` returns the expected count ``μ ≥ 0``; :meth:`evaluate`
    returns the **mean Poisson deviance** (the GLM-appropriate goodness-of-fit,
    lower is better), not MSE.

    Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        learning_rate: Gradient-descent step size (default ``0.01``).
        epochs: Passes over the data per :meth:`train` call (default ``50``).
        max_exponent: Clip on the linear predictor to avoid ``exp`` overflow
            (default ``30.0``).
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.01,
        epochs: int = 50,
        max_exponent: float = 30.0,
    ) -> None:
        self.input_dim = input_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.max_exponent = max_exponent
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0

    def _rate(self, X: np.ndarray) -> np.ndarray:
        """Expected count ``μ = exp(clip(w·x + b))``."""
        eta = np.clip(
            X @ self.weights + self.bias, -self.max_exponent, self.max_exponent
        )
        return np.exp(eta)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit by gradient descent on the Poisson negative log-likelihood.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Non-negative count targets of shape ``(n_samples,)``.
        """
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(self.epochs):
            mu = self._rate(X)
            err = mu - y
            self.weights -= self.lr * (X.T @ err) / n
            self.bias -= self.lr * float(np.sum(err)) / n

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
        """Return the expected count ``μ = exp(w·x + b)`` (non-negative).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted rates of shape ``(n_samples,)``.
        """
        return self._rate(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the mean Poisson deviance (lower is better).

        ``D = (2/n) Σ [ y·log(y/μ) - (y - μ) ]`` with the convention
        ``y·log(y/μ) = 0`` when ``y = 0``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True counts of shape ``(n_samples,)``.

        Returns:
            Mean Poisson deviance.

        Raises:
            ValueError: If *X* contains zero samples.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        mu = np.maximum(self._rate(X), 1e-12)
        # y*log(y/mu) -> 0 as y -> 0; guard the log there.
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        deviance = 2.0 * np.mean(term - (y - mu))
        return float(deviance)


class HuberRegressionModel(Model):
    """Robust linear regression with the Huber loss.

    Reference: Huber, "Robust Estimation of a Location Parameter", Annals of
    Mathematical Statistics 1964.

    Minimises ``Σ L_δ(yᵢ - (w·xᵢ + b))`` where the Huber loss is quadratic for
    small residuals and linear beyond a threshold ``δ``::

        L_δ(r) = ½r²              if |r| ≤ δ
               = δ(|r| - ½δ)      otherwise

    The linear tail caps the influence of large residuals, so a few grossly
    corrupted *target* values cannot dominate the fit the way they do under
    ordinary least squares.  This is a robustness axis orthogonal to L1/L2
    weight penalties (which regularise *coefficients*, not residuals).  Fitted
    by gradient descent; the gradient w.r.t. each prediction is the clipped
    residual ``clip(r, -δ, δ)``.

    Parameters are stored as ``[weights..., bias]``.

    Args:
        input_dim: Number of input features.
        delta: Huber threshold separating the quadratic and linear regimes
            (default ``1.0``).
        learning_rate: Gradient-descent step size (default ``0.01``).
        epochs: Passes over the data per :meth:`train` call (default ``100``).
    """

    def __init__(
        self,
        input_dim: int,
        delta: float = 1.0,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be > 0.")
        self.input_dim = input_dim
        self.delta = delta
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray = np.zeros(input_dim)
        self.bias: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit by gradient descent on the Huber loss.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
        """
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(self.epochs):
            r = y - (X @ self.weights + self.bias)
            # d L_δ / d prediction = -clip(r, -δ, δ).
            g = -np.clip(r, -self.delta, self.delta)
            self.weights -= self.lr * (X.T @ g) / n
            self.bias -= self.lr * float(np.sum(g)) / n

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
        """Compute linear predictions ``X·weights + bias``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """
        return X @ self.weights + self.bias
