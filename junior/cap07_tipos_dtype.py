"""Capítulo 7: Tipos de dados: dtype.

Parte: Básico.
Execute com: python cap07_tipos_dtype.py
"""


# === Os tipos mais comuns ===

import numpy as np

print(np.array([1, 2, 3]).dtype)
print(np.array([1.0, 2, 3]).dtype)
print(np.array([True, False]).dtype)
print(np.array([1 + 2j]).dtype)


# === Converter com astype ===

notas = np.array([7.9, 8.5, 9.99])
print(notas.astype(int))
print(np.round(notas).astype(int))


# === Quando o número não cabe ===

pequeno = np.array([250, 5], dtype=np.uint8)
print(pequeno + 10)

try:
    np.array([300], dtype=np.uint8)
except OverflowError as erro:
    print("OverflowError:", erro)


# === Precisão de decimais ===

print(0.1 + 0.2 == 0.3, np.isclose(0.1 + 0.2, 0.3))
print(np.finfo(np.float64).eps)
print(np.float32(16_777_216) + np.float32(1))


# === O que acontece quando tipos diferentes se encontram ===

a = np.array([1, 2, 3], dtype=np.int8)
print((a + 1).dtype)
print((a + np.int64(1)).dtype)
print((np.float32(3) + 3.0).dtype)
print(np.result_type(np.int8, np.float32), np.result_type(np.int64, np.float32))


# === Exercício: Economizar memória com o menor tipo possível ===

def economizar(idades):
    return idades.astype(np.min_scalar_type(idades.max()))


idades = np.array([34, 71, 12, 98, 45, 120], dtype=np.int64)
menor = economizar(idades)
assert menor.dtype == np.uint8
assert menor.nbytes < idades.nbytes
assert menor.tolist() == idades.tolist()
print("ok")
