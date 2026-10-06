"""Capítulo 23: Álgebra linear.

Parte: Intermediário.
Execute com: python cap23_algebra_linear.py
"""


# === Multiplicação de matrizes ===

import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([3.0, 5.0])
print(A @ A)
print(A.T @ b)
print(A * A)
print(np.array_equal(A * A, A @ A))

u = np.array([3.0, 4.0])
v = np.array([1.0, 0.0])
print(u @ v, np.linalg.norm(u), (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))


# === Resolver um sistema de equações ===

x = np.linalg.solve(A, b)
print(x, np.allclose(A @ x, b))


# === Determinante, inversa e autovalores ===

print(round(np.linalg.det(A), 6))
print(np.round(np.linalg.inv(A), 3))
valores, vetores = np.linalg.eigh(A)
print(np.round(valores, 3))


# === Ajustar uma reta: mínimos quadrados ===

rng = np.random.default_rng(1)
xs = np.linspace(0, 10, 50)
ys = 3 * xs + 2 + rng.normal(0, 0.5, 50)
X = np.column_stack([xs, np.ones_like(xs)])
(coef, intercepto), *_ = np.linalg.lstsq(X, ys, rcond=None)
print(abs(coef - 3) < 0.1, abs(intercepto - 2) < 0.5)


# === Lotes de matrizes ===

lote_a = np.ones((10, 3, 4))
lote_b = np.ones((10, 4, 2))
print((lote_a @ lote_b).shape)


# === Exercício: Projetar um vetor sobre outro ===

def projetar(u, v):
    return (u @ v) / (v @ v) * v


p = projetar(np.array([3.0, 4.0]), np.array([1.0, 0.0]))
assert np.allclose(p, [3.0, 0.0])
resto = np.array([3.0, 4.0]) - p
assert np.isclose(resto @ np.array([1.0, 0.0]), 0.0)
print("ok")
