from dataclasses import dataclass, field

import numpy as np

from rede_neural.otimizador import SGDMomento
from rede_neural.rede import RedeDensa
from rede_neural.tipos import Array, Inteiros


@dataclass
class Historico:
    perdas: list[float] = field(default_factory=list)
    acuracias: list[float] = field(default_factory=list)


def acuracia(rede: RedeDensa, X: Array, y: Inteiros) -> float:
    return float((rede.prever(X) == y).mean())


def treinar(
    rede: RedeDensa,
    X: Array,
    y: Inteiros,
    *,
    epocas: int,
    taxa: float = 0.1,
    momento: float = 0.9,
    tamanho_lote: int | None = None,
    rng: np.random.Generator | None = None,
) -> Historico:
    """Treina a rede. Sem ``tamanho_lote``, cada época é um único passo com todos os dados."""
    otimizador = SGDMomento(rede.parametros(), taxa, momento)
    rng = rng or np.random.default_rng(0)
    historico = Historico()
    for _ in range(epocas):
        if tamanho_lote is None:
            _, gradientes = rede.perda_e_gradientes(X, y)
            otimizador.passo(gradientes)
        else:
            ordem = rng.permutation(len(X))
            for inicio in range(0, len(X), tamanho_lote):
                lote = ordem[inicio : inicio + tamanho_lote]
                _, gradientes = rede.perda_e_gradientes(X[lote], y[lote])
                otimizador.passo(gradientes)
        perda, _ = rede.perda_e_gradientes(X, y)
        historico.perdas.append(perda)
        historico.acuracias.append(acuracia(rede, X, y))
    return historico
