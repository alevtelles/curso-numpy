import numpy as np

from rede_neural.dados import dividir, espiral, ou_exclusivo


def test_espiral_formas_e_classes() -> None:
    X, y = espiral(50, 3, 0.2, np.random.default_rng(0))
    assert X.shape == (150, 2) and y.shape == (150,)
    assert np.bincount(y).tolist() == [50, 50, 50]


def test_ou_exclusivo_rotulos() -> None:
    X, y = ou_exclusivo(300, np.random.default_rng(1))
    esperado = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    assert np.array_equal(y, esperado)
    assert set(np.unique(y)) == {0, 1}


def test_dividir_e_disjunto_e_embaralha() -> None:
    X, y = espiral(50, 3, 0.2, np.random.default_rng(0))
    Xt, Xv, yt, yv = dividir(X, y, 0.2, np.random.default_rng(2))
    assert len(Xt) == 120 and len(Xv) == 30
    assert len(np.unique(yv)) > 1  # o teste não é só de uma classe, porque embaralhou antes
