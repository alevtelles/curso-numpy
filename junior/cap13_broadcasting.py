"""Capítulo 13: Broadcasting.

Parte: Básico.
Execute com: python cap13_broadcasting.py
"""


# === A ideia: esticar, sem copiar ===

import numpy as np

grade = np.ones((3, 4))
linha = np.array([1, 2, 3, 4])
print(grade + linha)

coluna = np.array([[10], [20], [30]])
print(grade + coluna)
print((np.arange(3).reshape(3, 1) + np.arange(4)).shape)


# === A regra ===

def compativel(a, b):
    try:
        return np.broadcast_shapes(a, b)
    except ValueError:
        return None


print(compativel((3, 4), (4,)))
print(compativel((3, 1), (1, 4)))
print(compativel((5, 1, 3), (3,)))
print(compativel((3, 4), (3,)))

try:
    np.ones((3, 4)) + np.ones(3)
except ValueError as erro:
    print(erro)


# === Usos reais ===

notas = np.array([[7.0, 8.0, 9.0], [4.0, 6.0, 8.0]])
print(notas - notas.mean(axis=0))
print(notas / notas.sum(axis=1, keepdims=True))

indices = np.arange(4)
print(indices[:, None] - indices)


# === A memória não é copiada ===

esticado = np.broadcast_to(np.array([1, 2, 3]), (1000, 3))
print(esticado.shape, esticado.strides, esticado.flags["WRITEABLE"])


# === Exercício: Padronizar as colunas ===

def padronizar_colunas(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


dados = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
z = padronizar_colunas(dados)
assert np.allclose(z.mean(axis=0), 0)
assert np.allclose(z.std(axis=0), 1)
print("ok")
