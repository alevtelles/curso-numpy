"""Capítulo 27: Indexação avançada e malhas.

Parte: Intermediário.
Execute com: python cap27_indexacao_avancada.py
"""


# === Pegar o melhor de cada linha ===

import numpy as np

notas = np.array([[7, 9, 5], [4, 8, 6]])
melhor = notas.argmax(axis=1)
print(melhor, np.take_along_axis(notas, melhor[:, None], axis=1).ravel())
print(np.take_along_axis(notas, np.argsort(notas, axis=1), axis=1))


# === Selecionar linhas e colunas ao mesmo tempo ===

m = np.arange(16).reshape(4, 4)
print(m[np.ix_([0, 2], [1, 3])])


# === Malhas de coordenadas ===

x = np.linspace(-1, 1, 3)
y = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(x, y)
print(X)
print(np.hypot(X, Y).round(2))


# === Onde está o máximo? ===

plano = np.array([[3, 8, 1], [9, 2, 7]])
pos = np.unravel_index(plano.argmax(), plano.shape)
print(pos, plano[pos])
print(np.argwhere(plano > 6))


# === One-hot por indexação ===

rotulos = np.array([0, 2, 1])
print(np.eye(4)[rotulos])


# === Exercício: Ordenar as linhas pela soma ===

def ordenar_linhas_pela_soma(m):
    return m[np.argsort(m.sum(axis=1))]


m = np.array([[5, 5], [1, 1], [3, 2]])
assert ordenar_linhas_pela_soma(m).tolist() == [[1, 1], [3, 2], [5, 5]]
print("ok")
