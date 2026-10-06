"""Capítulo 9: Indexação booleana e fancy.

Parte: Básico.
Execute com: python cap09_indexacao_booleana.py
"""


# === Uma condição vira uma máscara ===

import numpy as np

notas = np.array([78, 92, 45, 88, 91, 30, 85])
mascara = notas > 75
print(mascara)
print(notas[mascara])
print(notas[notas > 75])
print((notas > 75).sum(), (notas > 75).mean().round(3))


# === Várias condições: `&`, `|` e `~` ===

presenca = np.array([95, 60, 80, 92, 70, 88, 97])
print(notas[(notas > 80) & (presenca > 90)])
print(notas[(notas < 50) | (presenca < 65)])
print(notas[~(notas > 80)])

try:
    notas[(notas > 80) and (presenca > 90)]
except ValueError as erro:
    print(erro)


# === Indexação fancy: uma lista de posições ===

nums = np.array([10, 20, 30, 40, 50])
print(nums[[0, 2, 4]])
print(nums[[4, 4, 0]])

tabela = np.arange(12).reshape(3, 4)
print(tabela[[0, 2], :])
print(tabela[[0, 1], [1, 3]])


# === Máscara e fancy devolvem cópias ===

recorte = nums[[0, 1]]
recorte[0] = 999
print(nums[0], np.shares_memory(nums, recorte))


# === Exercício: Aprovados e acima da média ===

def aprovados(notas, minimo=60):
    return notas[notas >= minimo]


def acima_da_media(notas):
    return np.nonzero(notas > notas.mean())[0]


turma = np.array([50, 90, 60, 40, 80])
assert aprovados(turma).tolist() == [90, 60, 80]
assert acima_da_media(turma).tolist() == [1, 4]
print("ok")
