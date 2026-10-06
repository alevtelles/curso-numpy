"""Capítulo 28: Janelas deslizantes e séries.

Parte: Intermediário.
Execute com: python cap28_janelas_series.py
"""


# === `sliding_window_view` ===

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

serie = np.array([1, 2, 3, 4, 5, 6], dtype=float)
janelas = sliding_window_view(serie, 3)
print(janelas)
print(janelas.mean(axis=1))
print(np.shares_memory(serie, janelas), janelas.flags["WRITEABLE"])


# === Duas outras formas de média móvel ===

def media_movel(x, k):
    acumulada = np.cumsum(np.insert(x, 0, 0.0))
    return (acumulada[k:] - acumulada[:-k]) / k


print(np.allclose(media_movel(serie, 3), janelas.mean(axis=1)))
print(np.convolve(serie, np.ones(3) / 3, mode="valid"))


# === Acumuladores: máximo corrido e queda máxima ===

precos = np.array([100, 110, 105, 120, 90, 95], dtype=float)
pico = np.maximum.accumulate(precos)
queda = (precos - pico) / pico
print(pico, queda.min().round(3))


# === Janelas em duas dimensões ===

img = np.arange(25, dtype=float).reshape(5, 5)
blocos = sliding_window_view(img, (3, 3))
print(blocos.shape, blocos.mean(axis=(-1, -2))[0, 0])


# === Exercício: Máximo móvel ===

def maximo_movel(x, k):
    return sliding_window_view(x, k).max(axis=1)


assert maximo_movel(np.array([1, 3, 2, 5, 4]), 2).tolist() == [3, 3, 5, 5]
assert maximo_movel(np.array([1, 3, 2, 5, 4]), 3).tolist() == [3, 5, 5]
print("ok")
