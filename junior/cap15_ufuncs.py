"""Capítulo 15: Funções universais (ufuncs).

Parte: Básico.
Execute com: python cap15_ufuncs.py
"""


# === Matemática sobre o array inteiro ===

import numpy as np

a = np.array([1.0, 4.0, 9.0])
print(np.sqrt(a), np.exp(np.array([0, 1])), np.log(np.array([1, np.e])))
print(np.add(a, 1), np.multiply(a, a))
print(np.abs(np.array([-3, 2])), np.round(np.array([1.234, 5.678]), 1))
print(np.clip(np.array([-5, 3, 12]), 0, 10))


# === Resultados impossíveis viram `nan` e `inf` ===

import warnings

with warnings.catch_warnings(record=True) as avisos:
    warnings.simplefilter("always")
    resultado = np.sqrt(np.array([4.0, -4.0]))
print(resultado, avisos[0].category.__name__)

with np.errstate(invalid="ignore", divide="ignore"):
    print(np.array([1.0, 0.0]) / np.array([0.0, 0.0]))


# === Escolher onde calcular: `where` e `out` ===

x = np.array([1.0, 4.0, 9.0, 16.0])
saida = np.zeros_like(x)
np.sqrt(x, out=saida, where=x > 5)
print(saida)


# === Divisão inteira e resto ===

print(np.array([7, -7]) // 2, np.mod(-7, 3))


# === Exercício: A função sigmoide ===

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


assert sigmoide(0) == 0.5
assert np.allclose(sigmoide(np.array([-1.0, 1.0])), [0.26894142, 0.73105858])
assert (np.diff(sigmoide(np.linspace(-5, 5, 20))) > 0).all()
print("ok")
