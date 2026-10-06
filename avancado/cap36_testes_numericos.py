"""Capítulo 36: Testes numéricos e estabilidade.

Parte: Avançado.
Execute com: python cap36_testes_numericos.py
"""


# === Igualdade com tolerância ===

import math

import numpy as np

print(0.1 + 0.2 == 0.3, np.isclose(0.1 + 0.2, 0.3), np.allclose([1.0, 2.0], [1.0 + 1e-9, 2.0]))
print(np.isclose(1e-10, 0.0), np.isclose(1e-10, 0.0, atol=0))


# === `numpy.testing` ===

from numpy.testing import assert_allclose

assert_allclose(np.array([1.0, 2.0]), np.array([1.0 + 1e-9, 2.0]), rtol=1e-6)
try:
    assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
except AssertionError as erro:
    print("Mismatched elements: 1 / 2 (50%)" in str(erro))


# === A ordem da soma muda o resultado ===

x = np.array([1e16, 1.0, -1e16])
print(x.sum(), x[0] + x[2] + x[1], math.fsum(x))

grande = np.ones(20_000_000, dtype=np.float32)
print(np.cumsum(grande)[-1], grande.sum())


# === Cancelamento catastrófico ===

y = 1e-10
print(np.log(1 + y), np.log1p(y))


# === Estabilidade numérica: o softmax ===

z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    ingenuo = np.exp(z) / np.exp(z).sum()
estavel = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print(ingenuo, estavel.round(4))


# === Verificar um gradiente por diferenças finitas ===

def f(w):
    return (w ** 3).sum() + 2 * w[0] * w[1]


def gradiente_analitico(w):
    g = 3 * w ** 2
    g[0] += 2 * w[1]
    g[1] += 2 * w[0]
    return g


def gradiente_numerico(funcao, w, h=1e-6):
    g = np.zeros_like(w)
    for i in range(w.size):
        mais = w.copy()
        mais[i] += h
        menos = w.copy()
        menos[i] -= h
        g[i] = (funcao(mais) - funcao(menos)) / (2 * h)
    return g


w = np.array([1.0, -2.0, 0.5])
print(np.allclose(gradiente_analitico(w), gradiente_numerico(f, w), atol=1e-5))


# === Exercício: Testar o softmax por propriedades ===

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


x = np.random.default_rng(0).normal(size=(4, 6))
assert_allclose(softmax(x).sum(axis=1), 1.0)
assert_allclose(softmax(x), softmax(x + 100.0))
assert_allclose(softmax(np.array([[1000.0, 1000.0]])), [[0.5, 0.5]])
print("ok")
