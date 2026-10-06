"""Capítulo 33: Arquivos maiores que a memória: memmap.

Parte: Avançado.
Execute com: python cap33_memmap.py
"""


# === Um arquivo que se comporta como array ===

from pathlib import Path

import numpy as np

caminho = Path("grande.bin")
mm = np.memmap(caminho, dtype=np.float32, mode="w+", shape=(1_000_000,))
mm[:] = np.arange(1_000_000, dtype=np.float32)
mm.flush()
print(caminho.stat().st_size, mm.shape)
del mm


# === Abrir só para ler ===

lido = np.memmap(caminho, dtype=np.float32, mode="r", shape=(1_000_000,))
print(lido[999_999], lido[:3], isinstance(lido, np.ndarray))
try:
    lido[0] = 1.0
except ValueError as erro:
    print(erro)


# === Processar em blocos ===

def media_em_blocos(mm, bloco=250_000):
    total = 0.0
    for inicio in range(0, len(mm), bloco):
        total += float(mm[inicio : inicio + bloco].sum(dtype=np.float64))
    return total / len(mm)


print(media_em_blocos(lido), np.arange(1_000_000, dtype=np.float64).mean())


# === `np.load` com `mmap_mode` ===

np.save("grande.npy", np.arange(1_000_000, dtype=np.float32))
vista = np.load("grande.npy", mmap_mode="r")
print(type(vista).__name__, vista[10])


# === Limpeza ===

del lido, vista
for nome in ("grande.bin", "grande.npy"):
    Path(nome).unlink()
print(Path("grande.bin").exists())


# === Exercício: Máximo de um arquivo, em blocos ===

import tempfile


def maximo_em_blocos(caminho, dtype, tamanho, bloco=1000):
    mm = np.memmap(caminho, dtype=dtype, mode="r", shape=(tamanho,))
    melhor = -np.inf
    for inicio in range(0, tamanho, bloco):
        melhor = max(melhor, float(mm[inicio : inicio + bloco].max()))
    del mm
    return melhor


with tempfile.TemporaryDirectory() as pasta:
    arquivo = Path(pasta) / "teste.bin"
    np.arange(5000, dtype=np.int32).tofile(arquivo)
    assert maximo_em_blocos(arquivo, np.int32, 5000, bloco=700) == 4999.0
print("ok")
