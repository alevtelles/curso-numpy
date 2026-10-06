"""Capítulo 6: Anatomia de um array.

Parte: Básico.
Execute com: python cap06_anatomia.py
"""


# === Os atributos ===

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape, arr.ndim, arr.size)
print(arr.dtype, arr.itemsize, arr.nbytes)
print(arr.T)
print(arr.strides)


# === Forma não é tamanho ===

print(len(arr), np.array([1, 2, 3]).shape, np.array(5).shape, np.array(5).ndim)


# === Exercício: Um resumo de qualquer array ===

def resumo(a):
    return {"forma": a.shape, "dimensoes": a.ndim, "tamanho": a.size, "bytes": a.nbytes}


r = resumo(np.zeros((4, 5)))
assert r == {"forma": (4, 5), "dimensoes": 2, "tamanho": 20, "bytes": 160}
print("ok")
