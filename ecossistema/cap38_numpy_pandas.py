"""Capítulo 38: NumPy e pandas.

Parte: Ecossistema.
Execute com: python cap38_numpy_pandas.py
"""


# === Uma tabela é um conjunto de arrays com nomes ===

import numpy as np
import pandas as pd

df = pd.DataFrame(
    {"nome": ["Ana", "Bia", "Caio", "Davi"], "nota": [7.5, 9.0, 6.0, 8.2], "faltas": [1, 0, 4, 2]}
)
print(df)
print(df.dtypes.astype(str).tolist())


# === Do pandas para o NumPy, e de volta ===

notas = df["nota"].to_numpy()
print(type(notas).__name__, notas.dtype, notas.mean().round(1))
print(df.to_numpy().dtype, df[["nota", "faltas"]].to_numpy().dtype, df[["nota", "faltas"]].to_numpy().shape)

matriz = np.array([[1.0, 2.0], [3.0, 4.0]])
print(pd.DataFrame(matriz, columns=["a", "b"]))


# === O array que volta é uma visão protegida ===

print(notas.flags["WRITEABLE"], notas.flags["OWNDATA"])
copia_editavel = df["nota"].to_numpy().copy()
copia_editavel[0] = 0.0
print(df["nota"].iloc[0])


# === A diferença mais perigosa: alinhamento por rótulo ===

s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
s2 = pd.Series([10, 20, 30], index=[2, 1, 0])
print((s1 + s2).tolist(), (s1.to_numpy() + s2.to_numpy()).tolist())


# === As ideias do NumPy, com outro nome ===

df["aprovado"] = np.where(df["nota"] >= 7, "sim", "não")
print(df["aprovado"].tolist(), df[df["nota"] > 7]["nome"].tolist())

vendas = pd.DataFrame(
    {"regiao": ["sul", "norte", "sul", "norte", "sul"], "valor": [10, 5, 20, 15, 30]}
)
print(vendas.groupby("regiao")["valor"].sum().to_dict())


# === Exercício: Normalizar uma coluna e devolver à tabela ===

def adicionar_zscore(df, coluna):
    resultado = df.copy()
    valores = df[coluna].to_numpy(dtype=float)
    resultado[coluna + "_z"] = (valores - valores.mean()) / valores.std()
    return resultado


original = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
novo = adicionar_zscore(original, "x")
assert list(original.columns) == ["x"]
assert list(novo.columns) == ["x", "x_z"]
assert np.isclose(novo["x_z"].mean(), 0)
assert np.isclose(novo["x_z"].std(ddof=0), 1)
print("ok")
