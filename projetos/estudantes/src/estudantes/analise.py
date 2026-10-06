import numpy as np

from estudantes.dados import Inteiros, Turma


def nivel_1(t: Turma) -> dict[str, float]:
    """Estatísticas descritivas das notas."""
    return {
        "maior": float(t.notas.max()),
        "menor": float(t.notas.min()),
        "media": float(t.notas.mean()),
        "mediana": float(np.median(t.notas)),
        "desvio_padrao": float(t.notas.std()),
        "variancia": float(t.notas.var()),
    }


def nivel_2(t: Turma) -> dict[str, Inteiros]:
    """Filtros por uma condição. Devolve os ids dos estudantes de cada grupo."""
    return {
        "nota_acima_de_90": t.ids[t.notas > 90],
        "nota_abaixo_de_40": t.ids[t.notas < 40],
        "presenca_acima_de_90": t.ids[t.presenca > 90],
        "estuda_mais_de_5h": t.ids[t.horas_estudo > 5],
        "mais_de_8_tarefas": t.ids[t.tarefas > 8],
    }


def nivel_3(t: Turma) -> dict[str, Inteiros]:
    """Filtros por várias condições. Cada condição vai entre parênteses, e o operador é & ou |."""
    return {
        "nota_alta_e_presenca_alta": t.ids[(t.notas > 80) & (t.presenca > 90)],
        "nota_baixa_e_presenca_baixa": t.ids[(t.notas < 40) & (t.presenca < 75)],
        "presenca_alta_nota_baixa": t.ids[(t.presenca > 90) & (t.notas < 60)],
        "pouco_estudo_nota_alta": t.ids[(t.horas_estudo < 2) & (t.notas > 80)],
    }


def correlacoes(t: Turma) -> dict[str, float]:
    """Correlação de cada variável com a nota (a primeira linha da matriz de correlação)."""
    variaveis = {
        "presenca": t.presenca,
        "horas_estudo": t.horas_estudo,
        "tarefas": t.tarefas.astype(np.float64),
    }
    matriz = np.corrcoef(np.vstack([t.notas, *variaveis.values()]))
    return {nome: float(matriz[0, i + 1]) for i, nome in enumerate(variaveis)}


def perfil_do_grupo(t: Turma, ids: Inteiros) -> dict[str, float]:
    """Média de cada atributo entre os estudantes do grupo."""
    mascara = np.isin(t.ids, ids)
    if not mascara.any():
        return {}
    return {
        "tamanho": float(mascara.sum()),
        "nota_media": float(t.notas[mascara].mean()),
        "presenca_media": float(t.presenca[mascara].mean()),
        "horas_estudo_medias": float(t.horas_estudo[mascara].mean()),
        "tarefas_medias": float(t.tarefas[mascara].mean()),
    }


def melhores(t: Turma, k: int = 5) -> Inteiros:
    """Ids dos k estudantes com as maiores notas, da maior para a menor."""
    return t.ids[np.argsort(-t.notas, kind="stable")[:k]]


def estudante(t: Turma, estudante_id: int) -> dict[str, float]:
    """Os dados de um estudante pelo id (os ids vão de 1 a n)."""
    posicoes = np.flatnonzero(t.ids == estudante_id)
    if posicoes.size == 0:
        raise KeyError(f"não existe o estudante {estudante_id}")
    i = int(posicoes[0])
    return {
        "id": float(t.ids[i]),
        "nota": float(t.notas[i]),
        "presenca": float(t.presenca[i]),
        "horas_estudo": float(t.horas_estudo[i]),
        "tarefas": float(t.tarefas[i]),
    }
