"""Capítulo 30: einsum e tensordot.

Parte: Avançado.
Execute com: python cap30_einsum.py
"""


# === A fórmula ===

import numpy as np

A = np.arange(6).reshape(2, 3)
B = np.arange(12).reshape(3, 4)
print(np.einsum("ij,jk->ik", A, B))
print(np.array_equal(np.einsum("ij,jk->ik", A, B), A @ B))


# === Um dicionário de padrões ===

M = np.arange(9).reshape(3, 3)
print(np.einsum("ii->", M), np.einsum("ii->i", M), np.einsum("ij->i", M), np.einsum("ij->j", M))

v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
print(np.einsum("i,i->", v, w), np.einsum("i,j->ij", v, w))


# === O caso de uso real: atenção em redes neurais ===

rng = np.random.default_rng(0)
Q = rng.normal(size=(2, 5, 8))
K = rng.normal(size=(2, 7, 8))
scores = np.einsum("bqd,bkd->bqk", Q, K)
print(scores.shape, np.allclose(scores, Q @ K.transpose(0, 2, 1)))


# === `tensordot` e a ordem das multiplicações ===

import timeit

print(np.allclose(np.tensordot(A, B, axes=1), A @ B))

C1 = rng.normal(size=(2000, 3))
C2 = rng.normal(size=(3, 2000))
C3 = rng.normal(size=(2000, 3))
esquerda = lambda: (C1 @ C2) @ C3
direita = lambda: C1 @ (C2 @ C3)
print(np.allclose(esquerda(), direita()))
t_esq = min(timeit.repeat(esquerda, number=1, repeat=3))
t_dir = min(timeit.repeat(direita, number=1, repeat=3))
print("a ordem importa:", t_esq > 3 * t_dir)


# === Exercício: Similaridade do cosseno entre todos os pares ===

def similaridade_cosseno(X):
    produtos = np.einsum("ik,jk->ij", X, X)
    normas = np.linalg.norm(X, axis=1)
    return produtos / np.outer(normas, normas)


X = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
S = similaridade_cosseno(X)
assert np.allclose(np.diag(S), 1.0)
assert np.allclose(S, S.T)
assert np.isclose(S[0, 1], 0.0)
assert np.isclose(S[0, 2], np.sqrt(2) / 2)
print("ok")
