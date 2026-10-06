"""Capítulo 24: Ler e gravar arrays.

Parte: Intermediário.
Execute com: python cap24_ler_gravar.py
"""


# === Texto: legível para pessoas e planilhas ===

from pathlib import Path

import numpy as np

dados = np.array([[1.5, 2.0], [3.25, 4.0]])
np.savetxt("tabela.csv", dados, delimiter=",", header="a,b", comments="", fmt="%.2f")
print(Path("tabela.csv").read_text(encoding="utf-8"))

lido = np.loadtxt("tabela.csv", delimiter=",", skiprows=1)
print(np.array_equal(lido, dados))


# === Binário: o formato `.npy` ===

np.save("matriz.npy", dados)
print(np.array_equal(np.load("matriz.npy"), dados))
print(Path("matriz.npy").stat().st_size > dados.nbytes)

np.savez("pacote.npz", x=dados, y=np.arange(3))
with np.load("pacote.npz") as pacote:
    print(sorted(pacote.files), pacote["y"])


# === Dados faltando em texto ===

Path("faltando.csv").write_text("a,b\n1,2\n3,\n5,6\n", encoding="utf-8")
cru = np.genfromtxt("faltando.csv", delimiter=",", skip_header=1)
print(cru)


# === Exercício: Gravar e conferir ===

import tempfile


def ida_e_volta(a, pasta):
    caminho = Path(pasta) / "teste.npy"
    np.save(caminho, a)
    lido = np.load(caminho)
    return lido.dtype == a.dtype and lido.shape == a.shape and np.array_equal(lido, a)


with tempfile.TemporaryDirectory() as pasta:
    assert ida_e_volta(np.arange(6, dtype=np.int16).reshape(2, 3), pasta)
    assert ida_e_volta(np.linspace(0, 1, 5), pasta)
print("ok")
