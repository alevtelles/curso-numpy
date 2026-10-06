"""Capítulo 29: Memória por dentro: strides, contiguidade e ordem.

Parte: Avançado.
Execute com: python cap29_memoria_strides.py
"""


# === O array é um bloco mais uma lista de `strides` ===

import numpy as np

a = np.arange(12, dtype=np.int64).reshape(3, 4)
print(a.strides, a.flags["C_CONTIGUOUS"], a.flags["F_CONTIGUOUS"])

t = a.T
print(t.strides, t.flags["C_CONTIGUOUS"], t.flags["F_CONTIGUOUS"], np.shares_memory(a, t))


# === Fatiar só muda os strides ===

print(a[:, ::2].strides, a[::2].strides, a[:, 0].strides)


# === Por que a contiguidade importa ===

import timeit

x = np.ones(16_000_000, dtype=np.float32)
contiguo = x[:1_000_000]
espacado = x[::16]
print(contiguo.size == espacado.size)
t_contiguo = min(timeit.repeat(lambda: contiguo.sum(), number=5, repeat=5))
t_espacado = min(timeit.repeat(lambda: espacado.sum(), number=5, repeat=5))
print("o contíguo foi mais rápido:", t_contiguo < t_espacado)


# === Reinterpretar bytes com `view` ===

inteiros = np.array([1, 256, 65536], dtype=np.int32)
print(inteiros.view(np.uint8))

bits = np.array([1.0], dtype=np.float32).view(np.uint32)[0]
print(hex(int(bits)))


# === `as_strided`: poder com perigo ===

from numpy.lib.stride_tricks import as_strided

y = np.arange(6)
janelas = as_strided(y, shape=(4, 3), strides=(y.strides[0], y.strides[0]))
print(janelas)


# === Exercício: Descrever a memória ===

def passos_em_bytes(a):
    return a.strides


def e_visao(a):
    return not a.flags["OWNDATA"]


z = np.zeros((2, 3), dtype=np.int16)
assert passos_em_bytes(z) == (6, 2)
assert not e_visao(z)
assert e_visao(z[:, 1:])
assert passos_em_bytes(z[:, ::2]) == (6, 4)
print("ok")
