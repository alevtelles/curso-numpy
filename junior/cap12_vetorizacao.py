"""Capítulo 12: Vetorização.

Parte: Básico.
Execute com: python cap12_vetorizacao.py
"""


# === O laço e a operação sobre o array ===

import time

import numpy as np

valores = np.array([1, 2, 3, 4, 5])
resultado = []
for x in valores:
    resultado.append(x * 2)
print(type(resultado).__name__, type(valores * 2).__name__)
print(valores * 2)


# === Fórmulas inteiras, sem laço ===

celsius = np.array([0.0, 25.0, 100.0])
print(celsius * 9 / 5 + 32)

precos = np.array([100.0, 250.0, 80.0])
quantidades = np.array([3, 1, 10])
print((precos * quantidades).sum())


# === Por que é mais rápido ===

def medir(funcao, repeticoes=3):
    melhor = float("inf")
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao()
        melhor = min(melhor, time.perf_counter() - inicio)
    return melhor


dados = np.arange(200_000, dtype=np.float64)


def com_laco(a):
    saida = np.empty_like(a)
    for i in range(a.size):
        saida[i] = a[i] ** 2 + 1
    return saida


def vetorizado(a):
    return a ** 2 + 1


print(np.array_equal(com_laco(dados), vetorizado(dados)))
mais_rapido = medir(lambda: com_laco(dados)) > 50 * medir(lambda: vetorizado(dados))
print("vetorizado pelo menos 50 vezes mais rápido:", mais_rapido)


# === O erro de usar funções que não são do NumPy ===

import math

try:
    math.sqrt(np.array([4.0, 9.0]))
except TypeError as erro:
    print("math.sqrt não aceita arrays:", type(erro).__name__)
print(np.sqrt(np.array([4.0, 9.0])))


# === Quando o laço é a resposta certa ===

juros = 0.01
saldo = 1000.0
saldos = []
for _ in range(3):
    saldo = saldo * (1 + juros) + 100
    saldos.append(round(saldo, 2))
print(saldos)


# === Exercício: Reescreva sem laço ===

def imc(pesos, alturas):
    return pesos / alturas ** 2


def soma_de_quadrados_pares(a):
    return (a[a % 2 == 0] ** 2).sum()


resultado = imc(np.array([70.0, 90.0]), np.array([1.75, 1.80]))
assert np.allclose(resultado, [22.857, 27.778], atol=0.001)
assert soma_de_quadrados_pares(np.arange(10)) == 120
print("ok")
