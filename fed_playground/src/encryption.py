"""Encryption scheme abstractions for federated learning.

Defines the :class:`EncryptionScheme` ABC and the :class:`NoEncryption` baseline
that passes model parameters through unchanged.  Custom schemes (e.g. additive
secret sharing, homomorphic encryption) can be added by subclassing
:class:`EncryptionScheme`.
"""

import abc
from typing import Any

import numpy as np


class EncryptionScheme(abc.ABC):
    """Abstract base class for encryption schemes used in federated learning.

    Every concrete scheme must implement :meth:`encrypt`, :meth:`decrypt`, and
    :meth:`aggregate`.  The aggregate operation is kept on the scheme because
    fully-homomorphic schemes must operate on ciphertexts, whereas plaintext
    schemes can delegate to the aggregation strategy.
    """

    @abc.abstractmethod
    def encrypt(self, params: np.ndarray) -> Any:
        """Encrypt model parameters before transmission.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            Encrypted representation of *params* (type depends on scheme).
        """

    @abc.abstractmethod
    def decrypt(self, encrypted_params: Any) -> np.ndarray:
        """Decrypt model parameters after reception.

        Args:
            encrypted_params: Encrypted parameters as returned by :meth:`encrypt`.

        Returns:
            Flat numpy array of decrypted parameters.
        """

    @abc.abstractmethod
    def aggregate(self, encrypted_params_list: list[Any]) -> Any:
        """Aggregate a list of (possibly encrypted) parameter vectors.

        For homomorphic schemes this operates on ciphertexts directly.
        For plaintext schemes it is typically a sum — division by N is handled
        by the :class:`~fed_playground.src.aggregation.AggregationStrategy`.

        Args:
            encrypted_params_list: Non-empty list of encrypted parameter vectors.

        Returns:
            Aggregated result in the same representation as the inputs.

        Raises:
            ValueError: If *encrypted_params_list* is empty.
        """


class GaussianDPEncryption(EncryptionScheme):
    """Differential privacy via Gaussian noise injection (local DP).

    Before transmission each party's parameters are perturbed with additive
    Gaussian noise ``N(0, σ²)``.  The noise is applied once at encrypt time;
    decryption is the identity.  Aggregation is element-wise summation.

    Choosing ``σ`` involves a privacy-utility trade-off: larger σ gives
    stronger (ε, δ)-DP guarantees but degrades model accuracy.

    Args:
        sigma: Standard deviation of the Gaussian noise (default ``0.1``).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, sigma: float = 0.1, seed: int | None = None) -> None:
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to *params*.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            Noisy parameter array of the same shape.
        """
        noise = self._rng.normal(0.0, self.sigma, size=params.shape)
        return params + noise

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        """Return *encrypted_params* unchanged (noise is not reversible).

        Args:
            encrypted_params: Noisy parameter array.

        Returns:
            The same array unmodified.
        """
        return encrypted_params

    def aggregate(self, encrypted_params_list: list[np.ndarray]) -> np.ndarray:
        """Sum noisy parameter vectors element-wise.

        Args:
            encrypted_params_list: Non-empty list of noisy parameter arrays.

        Returns:
            Element-wise sum.

        Raises:
            ValueError: If *encrypted_params_list* is empty.
        """
        if not encrypted_params_list:
            raise ValueError("encrypted_params_list must not be empty.")
        return sum(encrypted_params_list)  # type: ignore[return-value]


class AdditiveSecretSharing(EncryptionScheme):
    """Simplified additive secret-sharing for two-party aggregation.

    Each parameter vector is split into ``n_shares`` random additive shares
    that sum to the original.  A single :class:`AdditiveSecretSharing`
    instance is shared across all parties; each party's ``encrypt`` call
    creates new shares, retaining the *first* share locally and returning
    the *sum of all remaining shares* (a.k.a. the public share).

    This is a pedagogical approximation — a real protocol would route each
    share to a separate aggregation server.  Here the orchestrator receives
    the public share and reconstruction happens automatically in
    :meth:`decrypt`.

    Args:
        n_shares: Total number of shares per parameter (default ``2``).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, n_shares: int = 2, seed: int | None = None) -> None:
        if n_shares < 2:
            raise ValueError("n_shares must be at least 2.")
        self.n_shares = n_shares
        self._rng = np.random.default_rng(seed)

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        """Split *params* into additive shares and return their sum.

        Generates ``n_shares - 1`` random arrays and computes the last share
        as the residual so all shares sum to *params*.  Returns the element-
        wise sum of all shares (which equals *params*) to preserve aggregation
        semantics — the randomness is used only to simulate the split.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            Numpy array equal to *params* (shares cancel out on sum).
        """
        # Generate n-1 random shares; last share is the residual.
        random_shares = [
            self._rng.normal(0.0, 1.0, size=params.shape)
            for _ in range(self.n_shares - 1)
        ]
        last_share = params - sum(random_shares)
        all_shares = random_shares + [last_share]
        # The sum of all shares reconstructs the original.
        return sum(all_shares)  # type: ignore[return-value]

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        """Return *encrypted_params* unchanged (shares already reconstructed).

        Args:
            encrypted_params: Aggregated parameter array.

        Returns:
            The same array unmodified.
        """
        return encrypted_params

    def aggregate(self, encrypted_params_list: list[np.ndarray]) -> np.ndarray:
        """Sum reconstructed parameter vectors element-wise.

        Args:
            encrypted_params_list: Non-empty list of parameter arrays.

        Returns:
            Element-wise sum.

        Raises:
            ValueError: If *encrypted_params_list* is empty.
        """
        if not encrypted_params_list:
            raise ValueError("encrypted_params_list must not be empty.")
        return sum(encrypted_params_list)  # type: ignore[return-value]


class NoEncryption(EncryptionScheme):
    """Passthrough encryption scheme — no cryptographic operations applied.

    Useful as a baseline and for debugging.  Parameters are returned
    unchanged; aggregation is element-wise summation (division by N is
    applied by :class:`~fed_playground.src.aggregation.MeanAggregation`).
    """

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        """Return *params* unchanged.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            The same array unmodified.
        """
        return params

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        """Return *encrypted_params* unchanged.

        Args:
            encrypted_params: Numpy array (already plaintext).

        Returns:
            The same array unmodified.
        """
        return encrypted_params

    def aggregate(self, encrypted_params_list: list[np.ndarray]) -> np.ndarray:
        """Sum parameter vectors element-wise.

        Args:
            encrypted_params_list: Non-empty list of numpy parameter arrays.

        Returns:
            Element-wise sum of all arrays.

        Raises:
            ValueError: If *encrypted_params_list* is empty.
        """
        if not encrypted_params_list:
            raise ValueError("encrypted_params_list must not be empty.")
        return sum(encrypted_params_list)  # type: ignore[return-value]
