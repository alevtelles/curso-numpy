import numpy as np

from rede_neural.tipos import Array


class SGDMomento:
    """Gradiente descendente com momento: a velocidade acumula os passos anteriores."""

    def __init__(self, parametros: list[Array], taxa: float = 0.1, momento: float = 0.9) -> None:
        self.parametros = parametros
        self.taxa = taxa
        self.momento = momento
        self.velocidades = [np.zeros_like(p) for p in parametros]

    def passo(self, gradientes: list[Array]) -> None:
        for p, v, g in zip(self.parametros, self.velocidades, gradientes, strict=True):
            v *= self.momento
            v -= self.taxa * g
            p += v
