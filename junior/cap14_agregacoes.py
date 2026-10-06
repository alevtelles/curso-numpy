"""Capítulo 14: Agregações e o parâmetro axis.

Parte: Básico.
Execute com: python cap14_agregacoes.py
"""


# === As agregações mais usadas ===

import numpy as np

dados = np.array([[4, 9, 2], [11, 6, 15]])
print(dados.sum(), dados.mean(), np.median(dados))
print(dados.sum(axis=0), dados.sum(axis=1))
print(dados.max(axis=1), dados.argmax(), dados.argmax(axis=1))


# === Dispersão e posição ===

print(dados.std().round(3), dados.var().round(3), dados.std(ddof=1).round(3))
print(np.percentile(dados, [25, 50, 75]))


# === Mediana resiste a valores extremos, a média não ===

salarios = np.array([3000, 3200, 3100, 2900, 50000])
print(salarios.mean(), np.median(salarios))


# === Contar valores distintos ===

votos = np.array([2, 1, 2, 3, 2, 1])
valores, contagens = np.unique(votos, return_counts=True)
print(valores, contagens)


# === Agregar com `keepdims` ===

print(dados.sum(axis=1, keepdims=True))
print(dados / dados.sum(axis=1, keepdims=True))


# === Exercício: Um resumo por coluna ===

def resumo_por_coluna(dados):
    return {
        "minimo": dados.min(axis=0),
        "maximo": dados.max(axis=0),
        "media": dados.mean(axis=0),
    }


r = resumo_por_coluna(np.array([[4, 9, 2], [11, 6, 15]]))
assert r["minimo"].tolist() == [4, 6, 2]
assert r["maximo"].tolist() == [11, 9, 15]
assert r["media"].tolist() == [7.5, 7.5, 8.5]
print("ok")
