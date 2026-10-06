"""Capítulo 20: Ordenar, buscar e contar.

Parte: Intermediário.
Execute com: python cap20_ordenar_buscar.py
"""


# === `sort` e `argsort` ===

import numpy as np

v = np.array([30, 10, 50, 20, 40])
print(np.sort(v), v)
print(np.argsort(v))
print(v[np.argsort(v)[::-1]])

nomes = np.array(["Ana", "Bia", "Caio"])
notas = np.array([7.5, 9.0, 6.0])
print(nomes[np.argsort(-notas)])

m = np.array([[3, 1, 2], [9, 8, 7]])
print(np.sort(m, axis=1))
print(np.sort(m, axis=0))


# === Ordenar por vários critérios ===

sobrenome = np.array(["Silva", "Souza", "Silva", "Souza"])
idade = np.array([30, 25, 20, 40])
ordem = np.lexsort((idade, sobrenome))
print(ordem.tolist())


# === Buscar em um array ordenado ===

ordenado = np.array([10, 20, 30, 40])
print(np.searchsorted(ordenado, [5, 20, 25, 99]))

idades = np.array([5, 17, 18, 40, 65, 90])
faixas = np.digitize(idades, [18, 65])
print(faixas)


# === Valores únicos e pertencimento ===

a = np.array([3, 1, 2, 3, 1])
print(np.unique(a), np.isin(a, [1, 2]))
valores, primeira, contagem = np.unique(a, return_index=True, return_counts=True)
print(valores, primeira, contagem)


# === Os k maiores, sem ordenar tudo ===

pontos = np.array([55, 91, 12, 78, 66, 99, 40])
topo3 = np.argpartition(-pontos, 3)[:3]
print(sorted(pontos[topo3].tolist(), reverse=True))


# === Exercício: Ranking dos melhores ===

def ranking(nomes, notas, k):
    return nomes[np.argsort(-notas)[:k]]


nomes = np.array(["Ana", "Bia", "Caio", "Davi"])
notas = np.array([7.5, 9.0, 6.0, 8.2])
assert ranking(nomes, notas, 2).tolist() == ["Bia", "Davi"]
print("ok")
