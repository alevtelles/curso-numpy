"""Capítulo 19: Juntar, empilhar e dividir.

Parte: Intermediário.
Execute com: python cap19_juntar_dividir.py
"""


# === Juntar ao longo de um eixo ===

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(np.concatenate([a, b], axis=0))
print(np.concatenate([a, b.T], axis=1))

try:
    np.concatenate([a, b], axis=1)
except ValueError as erro:
    print(erro)


# === Empilhar em um eixo novo ===

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print(np.stack([x, y]).shape, np.stack([x, y], axis=1).shape)
print(np.vstack([x, y]), np.hstack([x, y]))


# === Dividir ===

dados = np.arange(10)
print(np.split(dados, 5))
print(np.array_split(dados, 3))
print(np.split(dados, [2, 7]))

rng = np.random.default_rng(0)
ordem = rng.permutation(10)
treino, teste = np.split(ordem, [8])
print(len(treino), len(teste), sorted(np.concatenate([treino, teste]).tolist()) == list(range(10)))


# === Repetir ===

print(np.repeat([1, 2], 3), np.tile([1, 2], 3))


# === O `np.append` em laço é uma armadilha ===

import timeit


def com_append(n):
    a = np.array([])
    for i in range(n):
        a = np.append(a, i)
    return a


def com_lista(n):
    return np.array([i for i in range(n)], dtype=float)


print(np.array_equal(com_append(500), com_lista(500)))
t_append = min(timeit.repeat(lambda: com_append(3000), number=1, repeat=3))
t_lista = min(timeit.repeat(lambda: com_lista(3000), number=1, repeat=3))
print("a lista foi pelo menos 5 vezes mais rápida:", t_append > 5 * t_lista)


# === Exercício: Adicionar a coluna de uns ===

def adicionar_vies(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


X = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
resultado = adicionar_vies(X)
assert resultado.shape == (3, 3)
assert (resultado[:, 0] == 1).all()
assert np.array_equal(resultado[:, 1:], X)
print("ok")
