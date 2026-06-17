"""Byzantine attack strategies for benchmarking robust aggregation.

An :class:`Attack` replaces the updates of the *byzantine* parties with
malicious ones, given the full set of (plaintext) party updates for the round.
Operating on the whole collection — not one party at a time — is what lets the
*omniscient* attacks (:class:`IPMAttack`, :class:`ALittleIsEnoughAttack`) craft
their payload from the honest parties' statistics, which is exactly how they
defeat distance/median-based defenses.

Attacks assume array-valued (plaintext) updates: they are meaningless over
additive-masking schemes, where individual shares carry no information.
"""

import abc

import numpy as np


class Attack(abc.ABC):
    """Abstract base class for Byzantine update-poisoning attacks."""

    @abc.abstractmethod
    def corrupt(
        self, updates: list[np.ndarray], byzantine_ids: list[int]
    ) -> list[np.ndarray]:
        """Return a new update list with the *byzantine_ids* entries poisoned.

        Args:
            updates: One plaintext parameter vector per party (honest values).
            byzantine_ids: Indices into *updates* controlled by the adversary.

        Returns:
            A new list; honest entries unchanged, byzantine entries replaced.
        """


class NoAttack(Attack):
    """Honest baseline — every party reports truthfully."""

    def corrupt(
        self, updates: list[np.ndarray], byzantine_ids: list[int]
    ) -> list[np.ndarray]:
        return list(updates)


class SignFlipAttack(Attack):
    """Byzantine parties send the negation of their own update, scaled.

    A classic model-poisoning attack (Blanchard et al., NeurIPS 2017): pushing
    in the opposite direction drags the FedAvg mean away from the optimum.

    Args:
        scale: Multiplier on the flipped update (default ``1.0``).
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def corrupt(
        self, updates: list[np.ndarray], byzantine_ids: list[int]
    ) -> list[np.ndarray]:
        out = list(updates)
        for i in byzantine_ids:
            out[i] = -self.scale * updates[i]
        return out


class GaussianAttack(Attack):
    """Byzantine parties send large random Gaussian noise vectors.

    The simplest untargeted attack (Blanchard et al., NeurIPS 2017).

    Args:
        sigma: Std-dev of the injected noise (default ``50.0``).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, sigma: float = 50.0, seed: int | None = None) -> None:
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def corrupt(
        self, updates: list[np.ndarray], byzantine_ids: list[int]
    ) -> list[np.ndarray]:
        out = list(updates)
        for i in byzantine_ids:
            out[i] = self._rng.normal(0.0, self.sigma, size=updates[i].shape)
        return out


class IPMAttack(Attack):
    """Inner-product manipulation: push against the honest mean.

    Reference: Xie, Koyejo & Gupta, "Fall of Empires: Breaking Byzantine-tolerant
    SGD by Inner Product Manipulation", UAI 2020.

    All byzantine parties send ``-epsilon · mean(honest updates)``, which keeps a
    negative inner product with the true gradient direction while staying small
    enough to evade norm/distance filters.

    Args:
        epsilon: Attack strength (default ``0.1``).
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon

    def corrupt(
        self, updates: list[np.ndarray], byzantine_ids: list[int]
    ) -> list[np.ndarray]:
        out = list(updates)
        byz = set(byzantine_ids)
        honest = [u for j, u in enumerate(updates) if j not in byz]
        if not honest:
            return out
        malicious = -self.epsilon * np.mean(honest, axis=0)
        for i in byzantine_ids:
            out[i] = malicious
        return out


class ALittleIsEnoughAttack(Attack):
    """Stay within the honest distribution to slip past robust defenses.

    Reference: Baruch, Baruch & Goldberg, "A Little Is Enough: Circumventing
    Defenses For Distributed Learning", NeurIPS 2019.

    Byzantine parties send ``mean(honest) - z · std(honest)`` coordinate-wise —
    a small, coordinated perturbation that median/Krum cannot distinguish from an
    honest update yet still biases the aggregate.

    Args:
        z: Std-dev multiplier controlling stealth vs. damage (default ``1.0``).
            ponytail: the paper derives z from (n, f) via a normal table; the
            constant default is the simple knob — set it per (n, f) for the
            optimal attack.
    """

    def __init__(self, z: float = 1.0) -> None:
        self.z = z

    def corrupt(
        self, updates: list[np.ndarray], byzantine_ids: list[int]
    ) -> list[np.ndarray]:
        out = list(updates)
        byz = set(byzantine_ids)
        honest = [u for j, u in enumerate(updates) if j not in byz]
        if not honest:
            return out
        stacked = np.stack(honest, axis=0)
        malicious = stacked.mean(axis=0) - self.z * stacked.std(axis=0)
        for i in byzantine_ids:
            out[i] = malicious
        return out


def _demo() -> None:
    """Self-check: every attack drags the FedAvg mean off the honest cluster."""
    rng = np.random.default_rng(0)
    honest = [np.ones(3) + 0.3 * rng.standard_normal(3) for _ in range(5)]
    updates = [*honest, np.ones(3)]  # 6th party will be byzantine
    byz = [5]
    clean_mean = np.mean(updates, axis=0)
    for atk in [
        SignFlipAttack(scale=10),
        GaussianAttack(sigma=50, seed=0),
        IPMAttack(epsilon=5),
        ALittleIsEnoughAttack(z=5),
    ]:
        poisoned = atk.corrupt(updates, byz)
        assert poisoned[0] is not None
        moved = np.linalg.norm(np.mean(poisoned, axis=0) - clean_mean)
        assert moved > 0.1, f"{type(atk).__name__} barely moved the mean"
    assert NoAttack().corrupt(updates, byz)[5] is updates[5]
    print("attacks self-check OK")


if __name__ == "__main__":
    _demo()
