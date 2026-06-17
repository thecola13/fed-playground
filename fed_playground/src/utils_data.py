"""Synthetic data generation and splitting utilities.

These helpers are used by :class:`~fed_playground.src.environment.Environment`
when no external :class:`~fed_playground.src.dataloader.DataLoader` is
provided, and by example scripts that need reproducible toy datasets.
"""

import numpy as np


def generate_linear_data(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    random_seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic linear regression dataset.

    Creates data according to ``y = X @ w + b + ε`` where *w* and *b* are
    drawn from a standard normal distribution and ``ε ~ N(0, noise)``.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of input features.
        noise: Standard deviation of the additive Gaussian noise.
        random_seed: Seed for ``numpy.random`` to ensure reproducibility.
            Pass ``None`` to use the current random state.

    Returns:
        Tuple ``(X, y)`` where *X* has shape ``(n_samples, n_features)``
        and *y* has shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(random_seed)
    X = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal(n_features)
    true_bias = float(rng.standard_normal())
    y = X @ true_weights + true_bias + rng.normal(0, noise, size=n_samples)
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    n_parties: int,
    random_seed: int | None = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a dataset uniformly and randomly among *n_parties* parties.

    The data is shuffled before splitting so that each party receives a
    representative random subset.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target vector of shape ``(n_samples,)``.
        n_parties: Number of parties to split the data across.
        random_seed: Seed for the shuffle permutation.

    Returns:
        List of ``n_parties`` ``(X_i, y_i)`` tuples with roughly equal sizes.

    Raises:
        ValueError: If *n_parties* is less than 1.
    """
    if n_parties < 1:
        raise ValueError(f"n_parties must be >= 1, got {n_parties}.")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    X_splits = np.array_split(X_shuffled, n_parties)
    y_splits = np.array_split(y_shuffled, n_parties)

    return list(zip(X_splits, y_splits, strict=True))
