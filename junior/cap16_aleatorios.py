"""Capítulo 16: Números aleatórios com Generator.

Parte: Básico.
Execute com: python cap16_aleatorios.py
"""


# === A forma moderna: `default_rng` ===

import numpy as np

rng = np.random.default_rng(42)
print(rng.random(3))
print(rng.integers(0, 10, size=5))
print(rng.normal(loc=0, scale=1, size=3))
print(rng.choice([10, 20, 30], size=2, replace=False))

a = np.random.default_rng(7).random(3)
b = np.random.default_rng(7).random(3)
print(np.array_equal(a, b))


# === Por que não `np.random.seed` ===

np.random.seed(42)
print(np.random.rand(3))

def sorteio(n, rng):
    return rng.integers(1, 7, size=n)


print(sorteio(5, np.random.default_rng(1)).tolist())


# === Embaralhar: no lugar ou em cópia ===

vetor = np.arange(1, 6)
resultado = rng.shuffle(vetor)
print(resultado, sorted(vetor.tolist()))

original = np.arange(1, 6)
embaralhado = rng.permutation(original)
print(original, sorted(embaralhado.tolist()))


# === Gerar um conjunto sintético ===

alturas = rng.normal(170, 10, size=1000)
print(abs(alturas.mean() - 170) < 1, abs(alturas.std() - 10) < 1)

sorteados = rng.choice(["cara", "coroa"], size=1000, p=[0.9, 0.1])
print((sorteados == "cara").mean() > 0.8)


# === Exercício: Simular dados ===

def simular_dados(n, rng):
    lancamentos = rng.integers(1, 7, size=n)
    return np.bincount(lancamentos, minlength=7)[1:]


contagens = simular_dados(6000, np.random.default_rng(0))
assert contagens.shape == (6,)
assert contagens.sum() == 6000
assert (contagens > 800).all()
print("ok")
