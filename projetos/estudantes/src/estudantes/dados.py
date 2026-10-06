from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Reais = npt.NDArray[np.float64]
Inteiros = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Turma:
    """Cinco arrays do mesmo tamanho: a posição i descreve sempre o mesmo estudante."""

    ids: Inteiros
    notas: Reais
    presenca: Reais
    horas_estudo: Reais
    tarefas: Inteiros

    def __post_init__(self) -> None:
        tamanhos = {len(self.ids), len(self.notas), len(self.presenca)}
        tamanhos |= {len(self.horas_estudo), len(self.tarefas)}
        if len(tamanhos) != 1:
            raise ValueError("todos os arrays da turma precisam ter o mesmo tamanho")

    def __len__(self) -> int:
        return len(self.ids)


def gerar_turma(n: int = 200, semente: int = 42) -> Turma:
    """Gera uma turma sintética em que a nota **depende** das outras variáveis.

    A nota é construída a partir da presença, das horas de estudo e das tarefas, mais um ruído,
    então os padrões que a análise encontra são reais, e não coincidência. Os coeficientes
    foram escolhidos para a nota média ficar perto de 63, com bastante dispersão: assim todos
    os grupos que o projeto pede (notas acima de 90, abaixo de 40, presença alta com nota
    baixa...) têm estudantes, e nenhuma pergunta termina com resposta zero.
    """
    rng = np.random.default_rng(semente)
    ids = np.arange(1, n + 1)
    presenca = np.clip(rng.normal(85, 10, n), 50, 100).round(1)
    horas_estudo = np.clip(rng.normal(4, 2, n), 0, 10).round(1)
    tarefas = rng.integers(0, 11, n)
    ruido = rng.normal(0, 10, n)
    notas = np.clip(18 + presenca * 0.25 + horas_estudo * 4.5 + tarefas * 1.2 + ruido, 0, 100)
    notas = notas.round(1)
    return Turma(ids, notas, presenca, horas_estudo, tarefas)
