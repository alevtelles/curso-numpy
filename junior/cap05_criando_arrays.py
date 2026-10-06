"""Capítulo 5: Criando arrays.

Parte: Básico.
Execute com: python cap05_criando_arrays.py
"""


# === A partir dos seus dados ===

import numpy as np

print(np.array([1, 2, 3]))
print(np.array([[1, 2], [3, 4]]))

try:
    np.array([[1, 2], [3]])
except ValueError as erro:
    print("erro de forma irregular:", "inhomogeneous" in str(erro))


# === Arrays de preenchimento: você dá a forma ===

print(np.zeros((2, 3)))
print(np.ones((2, 3), dtype=int))
print(np.full((2, 3), 7))

try:
    np.zeros(2, 3)
except TypeError as erro:
    print(erro)


# === Matrizes especiais ===

print(np.eye(3))
print(np.diag([1, 2, 3]))


# === Sequências ===

print(np.arange(0, 10, 2))
print(np.linspace(0, 1, 5))
print(np.logspace(0, 2, 3))

print(len(np.arange(0, 1, 0.1)), len(np.linspace(0, 1, 11)))


# === Arrays com a forma de outro ===

base = np.array([[1, 2, 3], [4, 5, 6]])
print(np.zeros_like(base))
print(np.ones_like(base, dtype=float).shape)


# === Exercício: Múltiplos e matrizes constantes ===

def multiplos_de(passo, limite):
    return np.arange(0, limite + 1, passo)


def matriz_constante(linhas, colunas, valor):
    return np.full((linhas, colunas), valor)


assert multiplos_de(5, 50).tolist() == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
assert matriz_constante(2, 3, 7).shape == (2, 3)
assert (matriz_constante(2, 3, 7) == 7).all()
print("ok")
