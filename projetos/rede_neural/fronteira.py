"""Desenha a fronteira de decisão que a rede aprendeu na espiral.

Precisa do matplotlib, que não é dependência do projeto:
    uv run --with matplotlib python fronteira.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rede_neural.dados import dividir, espiral
from rede_neural.rede import RedeDensa
from rede_neural.treino import treinar

rng = np.random.default_rng(0)
X, y = espiral(100, 3, 0.2, rng)
X_treino, X_teste, y_treino, y_teste = dividir(X, y, 0.2, rng)
rede = RedeDensa([2, 64, 64, 3], rng, l2=1e-4)
treinar(rede, X_treino, y_treino, epocas=1000)

# Uma malha de pontos cobrindo o plano: a rede classifica todos de uma vez (vetorizado).
xs = np.linspace(-1.2, 1.2, 300)
malha_x, malha_y = np.meshgrid(xs, xs)
pontos = np.column_stack([malha_x.ravel(), malha_y.ravel()])
classes = rede.prever(pontos).reshape(malha_x.shape)

fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.contourf(malha_x, malha_y, classes, alpha=0.35, levels=[-0.5, 0.5, 1.5, 2.5])
ax.scatter(X_treino[:, 0], X_treino[:, 1], c=y_treino, s=10, edgecolors="none")
ax.scatter(X_teste[:, 0], X_teste[:, 1], c=y_teste, s=28, edgecolors="black", linewidths=0.6)
ax.set_title("Fronteira aprendida (pontos com borda: teste)")
ax.set_aspect("equal")
fig.savefig("fronteira.png")
