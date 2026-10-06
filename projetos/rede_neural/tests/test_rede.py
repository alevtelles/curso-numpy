import numpy as np

from rede_neural.dados import dividir, espiral, ou_exclusivo
from rede_neural.otimizador import SGDMomento
from rede_neural.rede import RedeDensa, softmax
from rede_neural.tipos import Array
from rede_neural.treino import acuracia, treinar


def test_probabilidades_tem_a_forma_certa_e_somam_1() -> None:
    rede = RedeDensa([4, 6, 3], np.random.default_rng(0))
    p = rede.probabilidades(np.random.default_rng(1).normal(size=(5, 4)))
    assert p.shape == (5, 3)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_softmax_e_estavel_com_valores_enormes() -> None:
    p = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0)


def test_gradiente_confere_com_diferencas_finitas() -> None:
    """O teste mais importante: o gradiente escrito à mão bate com a derivada numérica."""
    rng = np.random.default_rng(3)
    rede = RedeDensa([3, 5, 4, 2], rng, l2=0.01)
    X = rng.normal(size=(6, 3))
    y = rng.integers(0, 2, size=6)
    _, analiticos = rede.perda_e_gradientes(X, y)

    h = 1e-5
    for parametro, analitico in zip(rede.parametros(), analiticos, strict=True):
        numerico = np.zeros_like(parametro)
        for indice in np.ndindex(parametro.shape):
            original = parametro[indice]
            parametro[indice] = original + h
            mais, _ = rede.perda_e_gradientes(X, y)
            parametro[indice] = original - h
            menos, _ = rede.perda_e_gradientes(X, y)
            parametro[indice] = original
            numerico[indice] = (mais - menos) / (2 * h)
        assert np.allclose(analitico, numerico, rtol=1e-4, atol=1e-7)


def test_otimizador_passo_simples_e_momento() -> None:
    p: Array = np.array([1.0])
    sgd = SGDMomento([p], taxa=0.1, momento=0.0)
    sgd.passo([np.array([2.0])])
    assert np.allclose(p, [0.8])

    q: Array = np.array([1.0])
    com_momento = SGDMomento([q], taxa=0.1, momento=0.5)
    com_momento.passo([np.array([2.0])])
    com_momento.passo([np.array([2.0])])
    assert np.allclose(q, [1.0 - 0.2 - 0.3])


def test_a_perda_cai_durante_o_treino() -> None:
    rng = np.random.default_rng(0)
    X, y = espiral(60, 3, 0.2, rng)
    rede = RedeDensa([2, 32, 3], rng)
    historico = treinar(rede, X, y, epocas=200)
    assert historico.perdas[-1] < 0.5 * historico.perdas[0]


def test_aprende_o_ou_exclusivo() -> None:
    rng = np.random.default_rng(0)
    X, y = ou_exclusivo(300, rng)
    Xt, Xv, yt, yv = dividir(X, y, 0.2, rng)
    rede = RedeDensa([2, 8, 2], rng, l2=1e-4)
    treinar(rede, Xt, yt, epocas=500)
    assert acuracia(rede, Xv, yv) >= 0.95


def test_resolve_a_espiral_em_dados_novos() -> None:
    rng = np.random.default_rng(0)
    X, y = espiral(100, 3, 0.2, rng)
    Xt, Xv, yt, yv = dividir(X, y, 0.2, rng)
    rede = RedeDensa([2, 64, 64, 3], rng, l2=1e-4)
    treinar(rede, Xt, yt, epocas=1000)
    assert acuracia(rede, Xt, yt) > 0.97
    assert acuracia(rede, Xv, yv) > 0.95


def test_mesma_semente_mesmo_resultado() -> None:
    def rodar() -> float:
        rng = np.random.default_rng(5)
        X, y = espiral(40, 3, 0.2, rng)
        rede = RedeDensa([2, 16, 3], rng)
        return treinar(rede, X, y, epocas=50).perdas[-1]

    assert rodar() == rodar()
