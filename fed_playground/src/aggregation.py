"""Aggregation strategies for the federated learning orchestrator.

Each strategy receives the list of (possibly encrypted) model updates collected
from all parties and returns a single aggregated result.  The result is stored
as the new global model in the :class:`~fed_playground.src.orchestrator.Orchestrator`.
"""

import abc
from typing import Any

import numpy as np

from .encryption import EncryptionScheme


class AggregationStrategy(abc.ABC):
    """Abstract base class for model aggregation strategies."""

    @abc.abstractmethod
    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Aggregate a list of party model updates into a single result.

        Args:
            encrypted_models: Non-empty list of (possibly encrypted) parameter
                vectors, one per registered party.
            encryption_scheme: The scheme in use; may be needed to perform
                homomorphic operations on ciphertexts.

        Returns:
            Aggregated model parameters in the same representation as the inputs
            (plaintext numpy array for :class:`~fed_playground.src.encryption.NoEncryption`,
            ciphertext otherwise).

        Raises:
            ValueError: If *encrypted_models* is empty.
        """


class TrimmedMeanAggregation(AggregationStrategy):
    """Byzantine-robust aggregation via coordinate-wise trimmed mean.

    For each parameter dimension, removes the ``trim_fraction`` lowest and
    highest values before computing the mean.  This provides robustness
    against a bounded fraction of malicious or corrupted party updates.

    Requires plaintext (numpy) parameters; FHE ciphertexts are not supported.

    Args:
        trim_fraction: Fraction of parties to trim from each tail (default ``0.1``).
            Must satisfy ``2 * trim_fraction < 1``.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5).")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the coordinate-wise trimmed mean of party updates.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme (used for summation
                to obtain plaintext arrays before trimming).

        Returns:
            Trimmed-mean parameter vector as a numpy array.

        Raises:
            ValueError: If *encrypted_models* is empty or parameters are not
                numpy arrays.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")

        # Collect plaintext arrays; decrypt if necessary.
        arrays = [
            encryption_scheme.decrypt(p)
            if not isinstance(p, np.ndarray)
            else p
            for p in encrypted_models
        ]

        n = len(arrays)
        k = int(np.floor(n * self.trim_fraction))
        stacked = np.stack(arrays, axis=0)  # (n_parties, n_params)

        if k == 0:
            return stacked.mean(axis=0)

        # Sort along the party axis and slice out the k lowest and k highest.
        stacked_sorted = np.sort(stacked, axis=0)
        trimmed = stacked_sorted[k : n - k]
        return trimmed.mean(axis=0)


class MedianAggregation(AggregationStrategy):
    """Byzantine-robust aggregation via coordinate-wise median.

    For each parameter dimension the median value across all party updates
    is selected.  This is tolerant of up to ``(n_parties - 1) / 2``
    arbitrarily corrupted parties.

    Requires plaintext (numpy) parameters; FHE ciphertexts are not supported.
    """

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the coordinate-wise median of party updates.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            Median parameter vector as a numpy array.

        Raises:
            ValueError: If *encrypted_models* is empty.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")

        arrays = [
            encryption_scheme.decrypt(p)
            if not isinstance(p, np.ndarray)
            else p
            for p in encrypted_models
        ]
        return np.median(np.stack(arrays, axis=0), axis=0)


class MeanAggregation(AggregationStrategy):
    """Federated averaging (FedAvg) — compute the mean of all party updates.

    When :class:`~fed_playground.src.encryption.NoEncryption` is used the
    result is the true arithmetic mean.

    For schemes that support only homomorphic *addition* (not scalar division),
    the aggregated result is the **sum** and the caller is responsible for
    dividing by the number of parties after decryption.  This is documented
    behaviour and a deliberate trade-off for FHE compatibility.
    """

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the mean (or sum for opaque ciphertexts) of party updates.

        Args:
            encrypted_models: Non-empty list of party parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            Arithmetic mean if parameters are numpy arrays; element-wise sum
            otherwise (for FHE ciphertexts that do not support scalar division).

        Raises:
            ValueError: If *encrypted_models* is empty.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")

        summed = encryption_scheme.aggregate(encrypted_models)

        # Divide only when we have a plaintext numpy array.
        # FHE ciphertexts cannot be divided here without the private key;
        # the receiving party must divide by N after decryption.
        if isinstance(summed, np.ndarray):
            return summed / len(encrypted_models)

        return summed
