import numpy as np

from rede_neural.tipos import Array, Inteiros


def espiral(
    n_por_classe: int, classes: int, ruido: float, rng: np.random.Generator
) -> tuple[Array, Inteiros]:
    """Braços de espiral entrelaçados: um problema que nenhuma reta separa."""
    raio = np.linspace(0.0, 1.0, n_por_classe)
    angulo = np.linspace(0.0, 4.0, n_por_classe)[None, :] + 4.0 * np.arange(classes)[:, None]
    angulo = angulo + rng.normal(0.0, ruido, size=(classes, n_por_classe))
    pontos = np.stack([raio * np.sin(angulo), raio * np.cos(angulo)], axis=-1)
    X: Array = pontos.reshape(-1, 2)
    y: Inteiros = np.repeat(np.arange(classes), n_por_classe)
    return X, y


def ou_exclusivo(n: int, rng: np.random.Generator) -> tuple[Array, Inteiros]:
    """Pontos em [-1, 1]²: a classe é 1 quando os sinais das coordenadas são diferentes."""
    X: Array = rng.uniform(-1.0, 1.0, size=(n, 2))
    y: Inteiros = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(np.int64)
    return X, y


def dividir(
    X: Array, y: Inteiros, fracao_teste: float, rng: np.random.Generator
) -> tuple[Array, Array, Inteiros, Inteiros]:
    """Separa treino e teste depois de embaralhar (os dados vêm ordenados por classe)."""
    ordem = rng.permutation(len(X))
    corte = int(len(X) * (1 - fracao_teste))
    treino, teste = ordem[:corte], ordem[corte:]
    return X[treino], X[teste], y[treino], y[teste]
