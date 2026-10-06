"""Capítulo 11: Reshape e transposição.

Parte: Básico.
Execute com: python cap11_reshape.py
"""


# === Remodelar sem mexer nos dados ===

import numpy as np

arr = np.arange(12)
print(arr.reshape(3, 4))
print(arr.reshape(2, -1).shape, arr.reshape(-1, 3).shape)
try:
    arr.reshape(5, 3)
except ValueError as erro:
    print(erro)


# === Transpor, achatar e inserir eixos ===

grade = np.arange(6).reshape(2, 3)
print(grade.T.shape, grade.transpose().shape)
print(grade.flatten(), grade.ravel())
print(grade[:, np.newaxis, :].shape, grade[..., None].shape)
print(np.squeeze(np.zeros((1, 3, 1))).shape)

r = grade.ravel()
r[0] = 99
print(grade[0, 0])
f = grade.flatten()
f[0] = -1
print(grade[0, 0])


# === O truque do eixo extra ===

vetor = np.array([1, 2, 3])
print(vetor.reshape(-1, 1))


# === Um caso real: lote de imagens ===

lote = np.zeros((10, 28, 28))
print(lote.reshape(10, -1).shape)


# === Exercício: Coluna e planilha ===

def para_coluna(v):
    return v.reshape(-1, 1)


def lote_para_planilha(imagens):
    return imagens.reshape(imagens.shape[0], -1)


assert para_coluna(np.array([1, 2, 3])).shape == (3, 1)
assert lote_para_planilha(np.zeros((10, 28, 28))).shape == (10, 784)
print("ok")
