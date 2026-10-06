"""Capítulo 4: Por que o NumPy é rápido.

Parte: Ambiente.
Execute com: python cap04_por_que_rapido.py
"""


# === Cada elemento de uma lista é um objeto ===

import sys

import numpy as np

lista = list(range(1_000))
array = np.arange(1_000)

print(type(lista[0]).__name__, "ocupa", sys.getsizeof(lista[0]), "bytes cada")
print(array.dtype, "ocupa", array.itemsize, "bytes por elemento")

bytes_lista = sys.getsizeof(lista) + sum(sys.getsizeof(x) for x in lista)
print("a lista gasta mais de 3 vezes a memória do array:", bytes_lista > 3 * array.nbytes)


# === Um tipo só, um bloco só ===

misturado = np.array([7, "a", 3.2, 9])
print(misturado.dtype, misturado)

print(array.flags["C_CONTIGUOUS"], array.strides)


# === Medir, em vez de acreditar ===

import time


def medir(funcao, repeticoes=5):
    melhor = float("inf")
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao()
        melhor = min(melhor, time.perf_counter() - inicio)
    return melhor


grande_lista = list(range(1_000_000))
grande_array = np.arange(1_000_000)
t_lista = medir(lambda: [x * 2 for x in grande_lista])
t_array = medir(lambda: grande_array * 2)
print("o array foi mais rápido:", t_array < t_lista)
print("pelo menos 10 vezes mais rápido:", t_lista > 10 * t_array)


# === Exercício: Quanto ocupa um array? ===

def bytes_do_array(n, dtype):
    return n * np.dtype(dtype).itemsize


assert bytes_do_array(1_000_000, "float64") == 8_000_000
assert bytes_do_array(1_000_000, "float32") == 4_000_000
assert bytes_do_array(1_000_000, "int8") == 1_000_000
print("ok")
