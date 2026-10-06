"""Capítulo 34: Ufuncs por dentro.

Parte: Avançado.
Execute com: python cap34_ufuncs_por_dentro.py
"""


# === Os métodos de uma ufunc ===

import numpy as np

v = np.array([1, 2, 3, 4])
print(np.add.reduce(v), np.multiply.reduce(v), np.add.accumulate(v), np.multiply.accumulate(v))
print(np.multiply.outer([1, 2, 3], [1, 2, 3]))


# === A armadilha dos índices repetidos ===

a = np.zeros(3, dtype=int)
indices = np.array([0, 0, 1, 2, 2, 2])
a[indices] += 1
print(a)

b = np.zeros(3, dtype=int)
np.add.at(b, indices, 1)
print(b)
print(np.bincount(indices, minlength=3))


# === `np.vectorize` não torna nada mais rápido ===

import timeit


def passos_collatz(n):
    passos = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        passos += 1
    return passos


vetorizada = np.vectorize(passos_collatz)
print(vetorizada(np.array([6, 7, 27])))

entradas = np.arange(1, 2000)
t_vec = min(timeit.repeat(lambda: vetorizada(entradas), number=1, repeat=3))
t_laco = min(timeit.repeat(lambda: [passos_collatz(int(n)) for n in entradas], number=1, repeat=3))
print("vectorize não é muito mais rápido que o laço:", t_vec > 0.5 * t_laco)


# === `__array_ufunc__`: como outras classes entram no jogo ===

class Embrulhado:
    def __init__(self, dados):
        self.dados = np.asarray(dados)

    def __array_ufunc__(self, ufunc, metodo, *entradas, **kwargs):
        abertas = [x.dados if isinstance(x, Embrulhado) else x for x in entradas]
        return Embrulhado(getattr(ufunc, metodo)(*abertas, **kwargs))


r = np.sqrt(Embrulhado([4, 9]))
print(type(r).__name__, r.dados)


# === Exercício: Contar por grupo ===

def contar_por_grupo(indices, n):
    contagem = np.zeros(n, dtype=int)
    np.add.at(contagem, indices, 1)
    return contagem


indices = np.array([3, 1, 3, 3, 0, 1])
assert contar_por_grupo(indices, 5).tolist() == [1, 2, 0, 3, 0]
assert np.array_equal(contar_por_grupo(indices, 5), np.bincount(indices, minlength=5))
print("ok")
