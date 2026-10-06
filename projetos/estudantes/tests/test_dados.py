import numpy as np
import pytest

from estudantes.dados import Turma, gerar_turma


def test_tamanho_e_intervalos() -> None:
    t = gerar_turma(200)
    assert len(t) == 200
    assert t.ids[0] == 1 and t.ids[-1] == 200
    assert t.presenca.min() >= 50 and t.presenca.max() <= 100
    assert t.horas_estudo.min() >= 0 and t.horas_estudo.max() <= 10
    assert t.tarefas.min() >= 0 and t.tarefas.max() <= 10
    assert t.notas.min() >= 0 and t.notas.max() <= 100


def test_mesma_semente_gera_a_mesma_turma() -> None:
    a, b = gerar_turma(50, semente=1), gerar_turma(50, semente=1)
    assert np.array_equal(a.notas, b.notas)
    assert not np.array_equal(a.notas, gerar_turma(50, semente=2).notas)


def test_arrays_de_tamanhos_diferentes_sao_recusados() -> None:
    t = gerar_turma(10)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        Turma(t.ids, t.notas[:5], t.presenca, t.horas_estudo, t.tarefas)
