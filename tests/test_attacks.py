"""Tests for fed_playground.src.attacks."""

import numpy as np

from fed_playground.src.attacks import (
    ALittleIsEnoughAttack,
    GaussianAttack,
    IPMAttack,
    NoAttack,
    SignFlipAttack,
)


def _updates(seed=0):
    rng = np.random.default_rng(seed)
    honest = [np.ones(4) + 0.2 * rng.standard_normal(4) for _ in range(5)]
    return [*honest, np.ones(4)], [5]  # last party is byzantine


def test_no_attack_is_identity():
    updates, byz = _updates()
    out = NoAttack().corrupt(updates, byz)
    for a, b in zip(updates, out, strict=False):
        np.testing.assert_array_equal(a, b)


def test_sign_flip_negates():
    updates, byz = _updates()
    out = SignFlipAttack(scale=2.0).corrupt(updates, byz)
    np.testing.assert_allclose(out[5], -2.0 * updates[5])


def test_every_attack_moves_the_mean():
    updates, byz = _updates()
    clean = np.mean(updates, axis=0)
    for atk in [
        SignFlipAttack(scale=5),
        GaussianAttack(sigma=20, seed=1),
        IPMAttack(epsilon=5),
        ALittleIsEnoughAttack(z=3),
    ]:
        poisoned = atk.corrupt(updates, byz)
        assert np.linalg.norm(np.mean(poisoned, axis=0) - clean) > 0.1


def test_ipm_pushes_against_honest_mean():
    updates, byz = _updates()
    honest_mean = np.mean(updates[:5], axis=0)
    out = IPMAttack(epsilon=0.5).corrupt(updates, byz)
    # Malicious update is anti-parallel to the honest mean.
    assert float(out[5] @ honest_mean) < 0


def test_attacks_do_not_mutate_input():
    updates, byz = _updates()
    snapshot = [u.copy() for u in updates]
    SignFlipAttack().corrupt(updates, byz)
    for a, b in zip(updates, snapshot, strict=False):
        np.testing.assert_array_equal(a, b)  # original list untouched


def test_omniscient_attacks_handle_all_byzantine():
    # No honest parties -> nothing to base the attack on -> updates unchanged.
    updates = [np.ones(3), np.ones(3)]
    out = IPMAttack().corrupt(updates, [0, 1])
    np.testing.assert_array_equal(out[0], np.ones(3))
