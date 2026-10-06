import numpy as np

from estudantes.analise import (
    correlacoes,
    estudante,
    melhores,
    nivel_1,
    nivel_2,
    nivel_3,
    perfil_do_grupo,
)
from estudantes.dados import gerar_turma

T = gerar_turma(200)


def test_nivel_1_e_coerente() -> None:
    r = nivel_1(T)
    assert r["menor"] <= r["mediana"] <= r["maior"]
    assert r["menor"] <= r["media"] <= r["maior"]
    assert np.isclose(r["desvio_padrao"] ** 2, r["variancia"])


def test_nivel_2_confere_com_contagem_por_laco() -> None:
    """A versão vetorizada precisa dar o mesmo resultado de um laço obviamente correto."""
    grupos = nivel_2(T)
    por_laco = [int(T.ids[i]) for i in range(len(T)) if T.notas[i] > 90]
    assert grupos["nota_acima_de_90"].tolist() == por_laco
    por_laco = [int(T.ids[i]) for i in range(len(T)) if T.tarefas[i] > 8]
    assert grupos["mais_de_8_tarefas"].tolist() == por_laco


def test_nivel_3_e_subconjunto_do_nivel_2() -> None:
    n2, n3 = nivel_2(T), nivel_3(T)
    assert set(n3["nota_alta_e_presenca_alta"]) <= set(n2["presenca_acima_de_90"])
    assert set(n3["presenca_alta_nota_baixa"]) <= set(n2["presenca_acima_de_90"])
    assert set(n3["nota_baixa_e_presenca_baixa"]) <= set(n2["nota_abaixo_de_40"])


def test_a_nota_correlaciona_mais_com_as_horas_de_estudo() -> None:
    c = correlacoes(T)
    assert c["horas_estudo"] > c["tarefas"] > c["presenca"] > 0


def test_os_grupos_pedidos_tem_estudantes() -> None:
    """Um projeto cuja resposta é sempre zero não ensina nada: a turma precisa ser variada."""
    excecao = "pouco_estudo_nota_alta"
    for nome, grupo in {**nivel_2(T), **nivel_3(T)}.items():
        if nome != excecao:
            assert len(grupo) > 0, nome


def test_estudar_pouco_e_tirar_nota_alta_e_raro() -> None:
    """Esse grupo pode ser vazio: as horas de estudo são o que mais pesa na nota."""
    n3 = nivel_3(T)
    assert len(n3["pouco_estudo_nota_alta"]) < len(n3["presenca_alta_nota_baixa"])


def test_perfil_do_grupo() -> None:
    ids = nivel_2(T)["nota_acima_de_90"]
    perfil = perfil_do_grupo(T, ids)
    assert perfil["tamanho"] == len(ids)
    assert perfil["nota_media"] > 90
    assert perfil_do_grupo(T, np.array([], dtype=np.int64)) == {}


def test_melhores_e_busca_por_id() -> None:
    topo = melhores(T, 3)
    notas_do_topo = [estudante(T, int(i))["nota"] for i in topo]
    assert notas_do_topo == sorted(notas_do_topo, reverse=True)
    assert notas_do_topo[0] == T.notas.max()
    assert estudante(T, 1)["id"] == 1
