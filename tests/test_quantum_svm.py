"""Quantum SVM tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_svm import (
    xor_dataset, two_moons_dataset, circles_dataset,
    QuantumSVM, train_quantum_svm,
    accuracy, train_test_split,
)


# ---- Datasets ----

def test_xor_dataset_shape():
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=5, rng=rng)
    # 4 clusters × 5 points = 20 samples, 2 features.
    assert X.shape == (20, 2)
    assert y.shape == (20,)
    assert set(np.unique(y)) == {-1, +1}


def test_two_moons_shape():
    rng = np.random.default_rng(0)
    X, y = two_moons_dataset(n_per_class=7, rng=rng)
    assert X.shape == (14, 2)
    assert set(np.unique(y)) == {-1, +1}


def test_circles_shape():
    rng = np.random.default_rng(0)
    X, y = circles_dataset(n_per_class=6, rng=rng)
    assert X.shape == (12, 2)


def test_xor_class_balance():
    """Each class has the same number of samples."""
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=5, rng=rng)
    assert sum(y == +1) == sum(y == -1)


# ---- Train/test split ----

def test_train_test_split_disjoint_indices():
    rng = np.random.default_rng(0)
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_frac=0.3, rng=rng)
    n_train = X_tr.shape[0]
    n_test = X_te.shape[0]
    assert n_train + n_test == 20
    # No shared values (using the index trick).
    assert set(X_tr.flatten()).isdisjoint(set(X_te.flatten()))


def test_train_test_split_fraction():
    rng = np.random.default_rng(0)
    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    _, _, X_te, _ = train_test_split(X, y, test_frac=0.4, rng=rng)
    assert X_te.shape[0] == 4


# ---- SVM training ----

def test_svm_training_returns_model():
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, rng=rng)
    model = train_quantum_svm(X, y, C=1.0, reps=2)
    assert isinstance(model, QuantumSVM)
    assert len(model.alpha) == len(y)
    assert len(model.support_idx) >= 1


def test_svm_alpha_bounds():
    """Each α should satisfy 0 ≤ α_i ≤ C."""
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, rng=rng)
    model = train_quantum_svm(X, y, C=1.5, reps=2)
    assert (model.alpha >= -1e-6).all()
    assert (model.alpha <= 1.5 + 1e-6).all()


def test_svm_dual_constraint():
    """sum_i α_i y_i should be ≈ 0."""
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, rng=rng)
    model = train_quantum_svm(X, y, C=1.0, reps=2)
    constraint = float(model.alpha @ model.y_train)
    assert abs(constraint) < 1e-4


def test_svm_rejects_non_binary_labels():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        train_quantum_svm(X, y)


def test_svm_rejects_bad_X_shape():
    with pytest.raises(ValueError):
        train_quantum_svm(np.zeros(10), np.ones(10))


# ---- Predictions ----

def test_svm_predict_returns_label():
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, rng=rng)
    model = train_quantum_svm(X, y, C=1.0, reps=2)
    pred = model.predict(X[0])
    assert pred in (-1, +1)


def test_svm_predict_is_deterministic():
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, rng=rng)
    model = train_quantum_svm(X, y, C=1.0, reps=2)
    x_test = np.array([0.5, 0.5])
    p1 = model.predict(x_test)
    p2 = model.predict(x_test)
    assert p1 == p2


def test_svm_decision_function_continuous():
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, rng=rng)
    model = train_quantum_svm(X, y, C=1.0, reps=2)
    # Decision function should vary smoothly.
    f1 = model.decision_function(np.array([1.0, 1.0]))
    f2 = model.decision_function(np.array([1.01, 1.0]))
    # Small input change → small output change.
    assert abs(f1 - f2) < 0.5


# ---- Train accuracy ----

def test_svm_xor_train_accuracy():
    """Quantum SVM should fit the training XOR data well."""
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=6, noise=0.1, rng=rng)
    model = train_quantum_svm(X, y, C=2.0, reps=2)
    acc = accuracy(model, X, y)
    assert acc > 0.85


def test_svm_circles_train_accuracy():
    """Quantum SVM should fit circles training data."""
    rng = np.random.default_rng(0)
    X, y = circles_dataset(n_per_class=6, noise=0.03, rng=rng)
    model = train_quantum_svm(X, y, C=2.0, reps=2)
    acc = accuracy(model, X, y)
    assert acc > 0.8


def test_accuracy_perfect_for_trivial_classifier():
    """Accuracy on training data should be ≥ a chance baseline."""
    rng = np.random.default_rng(0)
    X, y = xor_dataset(n_per_class=4, noise=0.05, rng=rng)
    model = train_quantum_svm(X, y, C=2.0, reps=2)
    # accuracy is at least 0.5 (random chance).
    assert accuracy(model, X, y) >= 0.5
