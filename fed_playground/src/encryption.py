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

    #: True for schemes whose individual ciphertexts are meaningless in
    #: isolation and only reconstruct under summation (e.g. additive masking).
    #: Such schemes support *only* linear aggregation — coordinate-wise order
    #: statistics (median, trimmed mean) cannot be computed over masked shares.
    is_linear_only: bool = False

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
    """Additive secret-sharing via masking, with masks that cancel on sum.

    Each party splits its update ``x_i`` into ``n_shares`` additive shares:
    ``n_shares - 1`` random *secret* shares plus one *public* share
    ``p_i = x_i - Σ(secret shares)``.  Only the public share leaves the party;
    the secret-share total is retained in the (shared) scheme instance, which
    plays the role of a *separate, non-colluding aggregation server*.

    Reconstruction happens **only in aggregate**: summing every party's public
    share and adding back the retained secret totals yields ``Σ x_i`` exactly,
    because each party's mask cancels against its own retained share::

        Σ p_i + Σ secret_i = Σ (x_i - secret_i) + Σ secret_i = Σ x_i

    Security claim — input-private against the orchestrator / aggregation
    strategy, which only ever observes public shares (each ``x_i`` masked by
    independent secret randomness it never sees).  Reconstruction succeeds only
    in aggregate, never for an individual party.  The protocol is **linear
    only**: order statistics (median, trimmed mean) cannot be computed over
    masked shares — use :class:`~fed_playground.src.aggregation.MeanAggregation`.

    Pedagogical caveat — in this single-process simulation the public-share
    aggregator and the secret-share holder live in the same object, so the
    *trust separation* is simulated, not enforced; the *information flow*
    (the aggregator never touches a plaintext update) is faithful.

    Args:
        n_shares: Additive shares per update; ``n_shares - 1`` are secret
            (default ``2`` → one secret mask, one public share).
        seed: Optional RNG seed for reproducibility.
    """

    is_linear_only = True

    def __init__(self, n_shares: int = 2, seed: int | None = None) -> None:
        if n_shares < 2:
            raise ValueError("n_shares must be at least 2.")
        self.n_shares = n_shares
        self._rng = np.random.default_rng(seed)
        # Running total of secret shares issued since the last aggregate().
        # ponytail: one aggregate() must consume each batch of encrypt() calls
        # (true for the Orchestrator); stale masks would corrupt the next sum.
        self._retained_total: np.ndarray | None = None

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        """Split *params* into additive shares; emit only the public share.

        The secret-share total is accumulated internally so it can be cancelled
        in :meth:`aggregate`.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            The public share ``params - Σ(secret shares)`` — statistically
            masked, meaningless in isolation.
        """
        secret_total = sum(
            self._rng.normal(0.0, 1.0, size=params.shape)
            for _ in range(self.n_shares - 1)
        )
        if self._retained_total is None:
            self._retained_total = secret_total
        else:
            self._retained_total = self._retained_total + secret_total
        return params - secret_total

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        """Return *encrypted_params* unchanged (reconstruction is in aggregate).

        Args:
            encrypted_params: A reconstructed/aggregated parameter array.

        Returns:
            The same array unmodified.
        """
        return encrypted_params

    def aggregate(self, encrypted_params_list: list[np.ndarray]) -> np.ndarray:
        """Reconstruct ``Σ x_i`` by summing public shares and cancelling masks.

        Adds the retained secret-share total back to the summed public shares,
        then clears the ledger for the next round.

        Args:
            encrypted_params_list: Non-empty list of public shares.

        Returns:
            Element-wise sum of the original (unmasked) updates.

        Raises:
            ValueError: If *encrypted_params_list* is empty.
        """
        if not encrypted_params_list:
            raise ValueError("encrypted_params_list must not be empty.")
        public_sum = sum(encrypted_params_list)
        if self._retained_total is not None:
            public_sum = public_sum + self._retained_total
            self._retained_total = None
        return public_sum  # type: ignore[return-value]


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


class LaplaceDPEncryption(EncryptionScheme):
    """Local differential privacy via the Laplace mechanism.

    Reference: Dwork, McSherry, Nissim & Smith, "Calibrating Noise to
    Sensitivity in Private Data Analysis", TCC 2006.

    Each party perturbs its update with additive Laplace noise of scale
    ``b = sensitivity / ε`` before transmission, giving pure ``ε``-differential
    privacy (no ``δ`` term, unlike the Gaussian mechanism).  The Laplace
    distribution's heavier tails are the price of that stronger guarantee.
    ``decrypt`` is the identity (noise is irreversible by design); aggregation
    is element-wise summation.

    The ``ε``-DP guarantee holds only if each party's update has L1-sensitivity
    bounded by *sensitivity* — i.e. updates are clipped to that L1 norm.  When
    *clip* is ``True`` this scheme enforces the bound; otherwise the caller is
    responsible and the stated ``ε`` is only nominal.

    Args:
        epsilon: Privacy budget ``ε`` > 0 (smaller = more private, more noise).
        sensitivity: L1-sensitivity bound of one update (default ``1.0``).
        clip: If ``True``, clip each update to L1 norm ``sensitivity`` so the
            DP guarantee is actually met (default ``False``, matching the
            plaintext-magnitude behaviour of :class:`GaussianDPEncryption`).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
        clip: bool = False,
        seed: int | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0.")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.clip = clip
        self.scale = sensitivity / epsilon
        self._rng = np.random.default_rng(seed)

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        """Clip (optionally) then add Laplace(0, sensitivity/ε) noise.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            Noisy parameter array of the same shape.
        """
        if self.clip:
            l1 = np.sum(np.abs(params))
            if l1 > self.sensitivity:
                params = params * (self.sensitivity / l1)
        noise = self._rng.laplace(0.0, self.scale, size=params.shape)
        return params + noise

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        """Return *encrypted_params* unchanged (noise is not reversible)."""
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


class PairwiseMaskingEncryption(EncryptionScheme):
    """Secure aggregation via pairwise additive masks that cancel on sum.

    Reference: Bonawitz et al., "Practical Secure Aggregation for
    Privacy-Preserving Machine Learning", ACM CCS 2017.

    Every unordered pair of parties ``{a, b}`` shares a per-round random mask
    ``p_ab``; party ``a`` adds ``+p_ab`` and party ``b`` adds ``-p_ab``.  Each
    emitted value ``x_i + Σ_j ±p_ij`` is masked into noise, yet **summing all
    parties' contributions cancels every pairwise term structurally**, exactly
    recovering ``Σ x_i`` — no party, and no reconstructor, can unmask an
    individual update.  This is the key advantage over a simple additive split:
    cancellation needs no trusted dealer.

    Security claim — input-private against the orchestrator / aggregation
    strategy, which observes only masked values; the sum reveals the aggregate
    and nothing else.  This is the linear-only regime (``is_linear_only``):
    order statistics cannot be computed over masks.

    Simulation caveats (``ponytail``: named ceilings):

    * Assumes exactly ``n_parties`` :meth:`encrypt` calls per round, in a stable
      order — true for the :class:`~fed_playground.src.orchestrator.Orchestrator`.
      Party index is inferred from the intra-round call counter.
    * Dropout recovery (the real protocol secret-shares each seed via Shamir so
      a dropped party's masks can be removed) is **not** simulated: a missing
      party in a round breaks cancellation.
    * The pairwise mask matrix is ``O(n²)`` vectors per round — fine for the
      handful of parties in a simulation.

    Args:
        n_parties: Number of participating parties (must match the simulation).
        mask_scale: Std-dev of each pairwise mask; larger ⇒ stronger statistical
            hiding (default ``10.0``).  Does not affect correctness.
        seed: Base RNG seed for reproducible pairwise masks (default ``0``).
    """

    is_linear_only = True

    def __init__(self, n_parties: int, mask_scale: float = 10.0, seed: int = 0) -> None:
        if n_parties < 1:
            raise ValueError("n_parties must be >= 1.")
        self.n_parties = n_parties
        self.mask_scale = mask_scale
        self.seed = seed
        self._call_count = 0  # advances once per encrypt(); index = count % n

    def _pair_mask(
        self, round_idx: int, a: int, b: int, shape: tuple[int, ...]
    ) -> np.ndarray:
        """Per-round mask for the unordered pair ``{a, b}`` (symmetric in a, b)."""
        lo, hi = (a, b) if a < b else (b, a)
        rng = np.random.default_rng([self.seed, round_idx, lo, hi])
        return rng.normal(0.0, self.mask_scale, size=shape)

    def _party_mask(self, round_idx: int, i: int, shape: tuple[int, ...]) -> np.ndarray:
        """Total antisymmetric mask added by party ``i``; Σ_i = 0 by construction."""
        mask = np.zeros(shape)
        for j in range(self.n_parties):
            if j == i:
                continue
            p = self._pair_mask(round_idx, i, j, shape)
            mask = mask + (p if i < j else -p)
        return mask

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        """Add this party's pairwise-mask total to *params*.

        The party index and round are inferred from the call counter.

        Args:
            params: Flat numpy array of model parameters.

        Returns:
            Masked parameter array; meaningless in isolation.
        """
        i = self._call_count % self.n_parties
        round_idx = self._call_count // self.n_parties
        self._call_count += 1
        return params + self._party_mask(round_idx, i, params.shape)

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        """Return *encrypted_params* unchanged (reconstruction is in aggregate)."""
        return encrypted_params

    def aggregate(self, encrypted_params_list: list[np.ndarray]) -> np.ndarray:
        """Sum masked updates; pairwise masks cancel, leaving ``Σ x_i``.

        Args:
            encrypted_params_list: Non-empty list of masked updates.

        Returns:
            Element-wise sum of the original (unmasked) updates.

        Raises:
            ValueError: If *encrypted_params_list* is empty.
        """
        if not encrypted_params_list:
            raise ValueError("encrypted_params_list must not be empty.")
        return sum(encrypted_params_list)  # type: ignore[return-value]
