"""Capítulo 22: Estatística: percentis, histogramas e correlação.

Parte: Intermediário.
Execute com: python cap22_estatistica.py
"""


# === Correlação entre duas variáveis ===

import numpy as np

rng = np.random.default_rng(42)
altura = rng.normal(170, 10, 500)
peso = 0.9 * altura - 85 + rng.normal(0, 5, 500)

matriz = np.corrcoef(altura, peso)
print(matriz.round(2))
print(0.8 < matriz[0, 1] < 0.95)


# === Histogramas ===

notas = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
contagens, limites = np.histogram(notas, bins=[0, 2, 4, 6])
print(contagens, limites)

contagens, limites = np.histogram(altura, bins=5)
print(contagens.sum(), len(limites))


# === Variações ao longo do tempo ===

precos = np.array([100.0, 102.0, 99.0, 105.0])
retornos = np.diff(precos) / precos[:-1]
print(retornos.round(4), np.cumsum([1, 2, 3, 4]))


# === Agrupar com `bincount` ===

categorias = np.array([0, 1, 1, 2, 2, 2])
valores = np.array([10, 20, 30, 1, 2, 3])
print(np.bincount(categorias), np.bincount(categorias, weights=valores))


# === Detectar valores atípicos (outliers) ===

dados = np.array([10, 12, 11, 13, 12, 95])
q1, q3 = np.percentile(dados, [25, 75])
iqr = q3 - q1
limite = q3 + 1.5 * iqr
print(dados[dados > limite])


# === Exercício: Detectar outliers dos dois lados ===

def outliers(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return x[(x < q1 - k * iqr) | (x > q3 + k * iqr)]


assert outliers(np.array([10, 12, 11, 13, 12, 95])).tolist() == [95]
assert outliers(np.array([-80, 10, 12, 11, 13, 12])).tolist() == [-80]
assert outliers(np.array([1, 2, 3, 4, 5])).tolist() == []
print("ok")
