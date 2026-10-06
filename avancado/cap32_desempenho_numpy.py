"""Capítulo 32: Desempenho: temporários, in-place e dtypes.

Parte: Avançado.
Execute com: python cap32_desempenho_numpy.py
"""


# === O custo dos temporários ===

import time
import tracemalloc

import numpy as np

rng = np.random.default_rng(0)
a = rng.random(2_000_000)
b = rng.random(2_000_000)


def com_temporarios(a, b):
    return (a * 2 + b * 3 - 1) ** 2


def sem_temporarios(a, b, t1, t2):
    np.multiply(a, 2, out=t1)
    np.multiply(b, 3, out=t2)
    t1 += t2
    t1 -= 1
    np.square(t1, out=t1)
    return t1


t1 = np.empty_like(a)
t2 = np.empty_like(a)
print(np.allclose(com_temporarios(a, b), sem_temporarios(a, b, t1, t2)))

def pico_de_memoria(funcao):
    tracemalloc.start()
    funcao()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return pico


pico_com = pico_de_memoria(lambda: com_temporarios(a, b))
pico_sem = pico_de_memoria(lambda: sem_temporarios(a, b, t1, t2))
print("o pico é menor sem temporários:", pico_sem < pico_com)
print("e muito menor, menos de 10% do outro:", pico_sem < 0.1 * pico_com)


# === Escolher o dtype certo ===

a32 = a.astype(np.float32)
print(a32.nbytes == a.nbytes // 2)
print(np.allclose(a32, a, atol=1e-6))


# === Iterar um array em Python é lento ===

def medir(funcao, repeticoes=3):
    melhor = float("inf")
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao()
        melhor = min(melhor, time.perf_counter() - inicio)
    return melhor


x = np.arange(100_000)
print("iterar a lista é mais rápido:", medir(lambda: [v for v in x.tolist()]) < medir(lambda: [v for v in x]))
print("sum do Python é pelo menos 10 vezes mais lento:", medir(lambda: sum(x)) > 10 * medir(lambda: x.sum()))


# === Exercício: Softmax estável e vetorizado ===

def softmax(x):
    deslocado = x - x.max(axis=-1, keepdims=True)
    e = np.exp(deslocado)
    return e / e.sum(axis=-1, keepdims=True)


x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
s = softmax(x)
assert np.allclose(s.sum(axis=1), 1.0)
assert np.isfinite(s).all()
assert np.allclose(s[0], s[1])
print("ok")
