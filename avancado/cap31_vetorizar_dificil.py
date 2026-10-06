"""Capítulo 31: Vetorizar problemas difíceis.

Parte: Avançado.
Execute com: python cap31_vetorizar_dificil.py
"""


# === Todas as distâncias entre pontos ===

import numpy as np

rng = np.random.default_rng(0)
pontos = rng.random((4, 2))

dif = pontos[:, None, :] - pontos[None, :, :]
dist = np.sqrt((dif ** 2).sum(axis=-1))
print(dist.shape, np.allclose(np.diag(dist), 0), np.allclose(dist, dist.T))

lento = np.array([[np.linalg.norm(p - q) for q in pontos] for p in pontos])
print(np.allclose(dist, lento))


# === O preço: o array intermediário ===

n = 10_000
print(n * n * 2 * 8 / 1e9, "GB")


# === Uma identidade que evita o intermediário ===

def distancias(X):
    q = (X ** 2).sum(axis=1)
    d2 = q[:, None] + q[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(d2, 0))


print(np.allclose(distancias(pontos), dist))


# === Quando ainda não cabe: processar em blocos ===

def mais_proximo(X, Y, bloco=500):
    resultado = np.empty(len(X), dtype=np.int64)
    for inicio in range(0, len(X), bloco):
        parte = X[inicio : inicio + bloco]
        d2 = ((parte[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
        resultado[inicio : inicio + bloco] = d2.argmin(axis=1)
    return resultado


X = rng.random((50, 3))
Y = rng.random((30, 3))
por_forca_bruta = np.array([np.argmin(((Y - x) ** 2).sum(axis=1)) for x in X])
print(np.array_equal(mais_proximo(X, Y, bloco=7), por_forca_bruta))


# === Agrupar sem laço: `reduceat` ===

grupos = np.array([2, 0, 1, 0, 2, 1])
valores = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
ordem = np.argsort(grupos, kind="stable")
g = grupos[ordem]
v = valores[ordem]
inicios = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
print(g[inicios], np.add.reduceat(v, inicios))


# === Exercício: Os k maiores de cada linha ===

def top_k_por_linha(M, k):
    candidatos = np.argpartition(-M, k - 1, axis=1)[:, :k]
    valores = np.take_along_axis(M, candidatos, axis=1)
    ordem = np.argsort(-valores, axis=1)
    return np.take_along_axis(candidatos, ordem, axis=1)


M = np.array([[5, 1, 9, 3], [2, 8, 4, 7]])
assert top_k_por_linha(M, 2).tolist() == [[2, 0], [1, 3]]
print("ok")
