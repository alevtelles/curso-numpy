"""Capítulo 35: Aleatoriedade avançada.

Parte: Avançado.
Execute com: python cap35_aleatoriedade_avancada.py
"""


# === Fluxos independentes com `SeedSequence` ===

import numpy as np

raiz = np.random.SeedSequence(2026)
geradores = [np.random.default_rng(filho) for filho in raiz.spawn(3)]
amostras = [g.random(2000) for g in geradores]
print(len(amostras), abs(np.corrcoef(amostras)[0, 1]) < 0.1)

a = np.random.default_rng(np.random.SeedSequence(2026).spawn(3)[1]).random(2)
b = np.random.default_rng(np.random.SeedSequence(2026).spawn(3)[1]).random(2)
print(np.array_equal(a, b))

filhos = np.random.default_rng(1).spawn(2)
print(type(filhos[0]).__name__)


# === Guardar e restaurar o estado ===

rng = np.random.default_rng(1)
estado = rng.bit_generator.state
primeiro = rng.random(2)
rng.bit_generator.state = estado
print(np.array_equal(primeiro, rng.random(2)))


# === Simulação de Monte Carlo ===

import math


def estimar_pi(n, rng):
    xy = rng.random((n, 2))
    dentro = (xy ** 2).sum(axis=1) <= 1
    return 4 * dentro.mean()


estimativa = estimar_pi(1_000_000, np.random.default_rng(0))
print(abs(estimativa - math.pi) < 0.01)

def dispersao(n, repeticoes=30):
    rng = np.random.default_rng(3)
    return np.std([estimar_pi(n, rng) for _ in range(repeticoes)])


print("mais amostras, menos variação:", dispersao(100_000) < dispersao(1_000))


# === Bootstrap: incerteza sem fórmula ===

rng = np.random.default_rng(5)
amostra = rng.normal(100, 15, 200)
medias = rng.choice(amostra, size=(2000, amostra.size), replace=True).mean(axis=1)
baixo, alto = np.percentile(medias, [2.5, 97.5])
print(baixo < amostra.mean() < alto, alto - baixo < 10)


# === Amostrar de uma distribuição qualquer ===

lam = 2.0
x = -np.log(1 - rng.random(100_000)) / lam
print(abs(x.mean() - 1 / lam) < 0.01)


# === Exercício: Intervalo de confiança por bootstrap ===

def intervalo_bootstrap(x, rng, n=1000, nivel=0.95):
    medias = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    cauda = (1 - nivel) / 2 * 100
    return tuple(np.percentile(medias, [cauda, 100 - cauda]))


dados = np.random.default_rng(9).normal(50, 10, 300)
baixo, alto = intervalo_bootstrap(dados, np.random.default_rng(10))
assert baixo < dados.mean() < alto
assert alto - baixo < 5
print("ok")
