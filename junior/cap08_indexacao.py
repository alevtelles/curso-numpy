"""Capítulo 8: Indexação e fatiamento.

Parte: Básico.
Execute com: python cap08_indexacao.py
"""


# === Um elemento, uma linha, uma coluna ===

import numpy as np

grade = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(grade[1, 2])
print(grade[:, 1])
print(grade[0, ::-1])
print(grade[1:, :2])
print(grade[-1])


# === Fatias: `inicio:fim:passo` ===

print(np.arange(10)[::3])


# === Atribuir a uma fatia ===

copia = grade.copy()
copia[0, :] = 0
copia[:, 2] += 100
print(copia)


# === Três dimensões e a reticência ===

cubo = np.arange(24).reshape(2, 3, 4)
print(cubo[1, 2, 3], cubo[:, 0, :].shape, cubo[..., 0].shape)


# === Índice fora dos limites ===

try:
    grade[5]
except IndexError as erro:
    print(erro)


# === Exercício: O miolo de uma matriz ===

def centro(a):
    return a[1:-1, 1:-1]


quadro = np.arange(16).reshape(4, 4)
assert np.array_equal(centro(quadro), np.array([[5, 6], [9, 10]]))
print("ok")
