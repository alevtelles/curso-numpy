"""Capítulo 25: Dados ausentes: NaN e arrays mascarados.

Parte: Intermediário.
Execute com: python cap25_dados_ausentes.py
"""


# === O `nan` contamina ===

import numpy as np

x = np.array([1.0, np.nan, 3.0, 4.0])
print(x.sum(), np.nansum(x), np.nanmean(x), np.nanmax(x))


# === Como detectar (e a pegadinha da igualdade) ===

print(np.isnan(x), np.isnan(x).sum())
print(np.nan == np.nan, np.isnan(np.nan))


# === Remover ou preencher ===

limpo = x[~np.isnan(x)]
preenchido = np.where(np.isnan(x), np.nanmean(x), x)
print(limpo, preenchido.round(2))
print(x > 2)


# === Só decimais têm `nan` ===

print(np.array([1, 2, np.nan]).dtype)
try:
    np.array([1, 2, 3])[0] = np.nan
except ValueError as erro:
    print(erro)


# === Arrays mascarados ===

m = np.ma.masked_invalid(x)
print(m, m.mean(), m.mask)

sentinela = np.ma.masked_equal([1, -999, 3], -999)
print(sentinela, sentinela.mean())


# === Exercício: Média por coluna ignorando buracos ===

def media_por_coluna(m):
    return np.nanmean(m, axis=0)


dados = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
assert media_por_coluna(dados).tolist() == [3.0, 6.0]
print("ok")
