"""Capítulo 26: Tipos estruturados e datas.

Parte: Intermediário.
Execute com: python cap26_estruturados_datas.py
"""


# === Registros com campos ===

import numpy as np

pessoas = np.array(
    [("Ana", 30, 1.65), ("Bia", 25, 1.70)],
    dtype=[("nome", "U10"), ("idade", "i4"), ("altura", "f8")],
)
print(pessoas["nome"], pessoas["idade"].mean())
print(pessoas[pessoas["idade"] > 26])
print(pessoas.dtype.names, pessoas.itemsize)
print(np.sort(pessoas, order="idade")["nome"])


# === Datas e durações ===

d = np.array(["2026-01-15", "2026-02-20"], dtype="datetime64[D]")
print(d, d.dtype)
print(d[1] - d[0])
print(np.datetime64("2026-03-01") + np.timedelta64(30, "D"))


# === Sequências de datas e conversão de unidade ===

print(np.arange("2026-01-01", "2026-01-06", dtype="datetime64[D]"))
print(d.astype("datetime64[M]"))


# === Dias úteis ===

print(np.busday_count("2026-10-01", "2026-10-31"))


# === Exercício: Dias entre duas datas ===

def dias_entre(inicio, fim):
    return int((np.datetime64(fim) - np.datetime64(inicio)).astype(int))


assert dias_entre("2026-01-01", "2026-10-06") == 278
assert dias_entre("2026-10-06", "2026-10-06") == 0
print("ok")
