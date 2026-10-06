"""Capítulo 10: Cópias e views.

Parte: Básico.
Execute com: python cap10_copias_views.py
"""


# === Uma view compartilha a memória ===

import numpy as np

original = np.arange(6)
fatia = original[1:4]
fatia[0] = 99
print(original)
print(np.shares_memory(original, fatia), fatia.base is original)


# === Quando você quer um array independente ===

independente = original[1:4].copy()
independente[0] = -1
print(original[1], np.shares_memory(original, independente), independente.base is None)


# === O mesmo objeto, dois nomes ===

a = np.array([1, 2, 3])
b = a
b += 10
print(a)
b = b + 10
print(a, b)


# === Funções que alteram o que recebem ===

def zerar_negativos(a):
    a[a < 0] = 0
    return a


dados = np.array([3, -1, 4, -5])
resultado = zerar_negativos(dados)
print(dados, resultado is dados)


# === `asarray` e `array` ===

x = np.arange(3)
print(np.asarray(x) is x, np.array(x) is x)


# === Exercício: Normalizar sem mutar a entrada ===

def normalizar(a):
    return (a - a.min()) / (a.max() - a.min())


entrada = np.array([10.0, 20.0, 30.0])
copia_da_entrada = entrada.copy()
saida = normalizar(entrada)
assert np.array_equal(entrada, copia_da_entrada)
assert saida.tolist() == [0.0, 0.5, 1.0]
assert not np.shares_memory(entrada, saida)
print("ok")
