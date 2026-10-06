"""Capítulo 39: NumPy e Matplotlib.

Parte: Ecossistema.
Execute com: python cap39_numpy_matplotlib.py
"""


# === Desenhar sem abrir janela ===

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
ax.plot(x, np.sin(x), label="seno")
ax.plot(x, np.cos(x), label="cosseno")
ax.legend()
fig.savefig("ondas.png")
plt.close(fig)


# === O gráfico salvo é um tensor ===

imagem = plt.imread("ondas.png")
print(imagem.shape, imagem.dtype)


# === O `hist` usa o `np.histogram` ===

dados = np.random.default_rng(0).normal(size=1000)
fig, ax = plt.subplots()
contagens, limites, _ = ax.hist(dados, bins=10)
esperado, _ = np.histogram(dados, bins=10)
print(np.array_equal(contagens, esperado))
plt.close(fig)


# === Um array 2D como imagem ===

X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
Z = np.exp(-(X ** 2 + Y ** 2))

fig, ax = plt.subplots()
imagem_z = ax.imshow(Z, extent=(-2, 2, -2, 2), origin="lower")
fig.colorbar(imagem_z)
fig.savefig("montanha.png")
plt.close(fig)
print(Z.shape, round(float(Z.max()), 2))


# === Gravar uma matriz direto como imagem ===

plt.imsave("cinza.png", Z, cmap="gray")
print(plt.imread("cinza.png").shape)


# === Limpeza ===

from pathlib import Path

for nome in ("ondas.png", "montanha.png", "cinza.png"):
    Path(nome).unlink()
print(Path("ondas.png").exists())


# === Exercício: Salvar um gráfico e conferir ===

import tempfile


def salvar_grafico(x, y, caminho):
    fig, ax = plt.subplots(dpi=100)
    ax.plot(x, y)
    fig.savefig(caminho)
    plt.close(fig)
    return plt.imread(caminho).shape


with tempfile.TemporaryDirectory() as pasta:
    forma = salvar_grafico(np.arange(5), np.arange(5) ** 2, Path(pasta) / "g.png")
assert forma == (480, 640, 4)
print("ok")
