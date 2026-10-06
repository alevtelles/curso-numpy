"""Capítulo 18: Medir desempenho de verdade.

Parte: Básico.
Execute com: python cap18_medir_desempenho.py
"""


# === O `timeit` e o melhor de vários ===

import timeit

import numpy as np

a = np.arange(1_000_000)
tempos = timeit.repeat(lambda: a.sum(), number=20, repeat=5)
print("melhor tempo por chamada abaixo de 10 ms:", min(tempos) / 20 < 0.01)


# === Quanto custa começar: tamanhos pequenos não contam a história ===

def razao(n):
    lista = list(range(n))
    array = np.arange(n)
    t_lista = min(timeit.repeat(lambda: [x * 2 for x in lista], number=50, repeat=5))
    t_array = min(timeit.repeat(lambda: array * 2, number=50, repeat=5))
    return t_lista / t_array


print("a vantagem cresce com o tamanho:", razao(100_000) > razao(100))
print("em 100 mil elementos o array ganha:", razao(100_000) > 5)


# === Medir memória ===

import tracemalloc

tracemalloc.start()
x = np.zeros(1_000_000)
atual, pico = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(round(atual / 1e6, 1), "MB")


# === Exercício: Comparar duas implementações ===

def comparar(f, g, repeticoes=5):
    t_f = min(timeit.repeat(f, number=10, repeat=repeticoes))
    t_g = min(timeit.repeat(g, number=10, repeat=repeticoes))
    return t_f / t_g


razao_medida = comparar(lambda: sum(range(10000)), lambda: np.arange(10000).sum())
assert razao_medida > 1
print("ok")
