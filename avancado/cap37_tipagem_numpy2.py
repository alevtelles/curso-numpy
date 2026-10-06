"""Capítulo 37: Tipagem, API e o NumPy 2.

Parte: Avançado.
Execute com: python cap37_tipagem_numpy2.py
"""


# === Anotar funções que usam arrays ===

import numpy as np
import numpy.typing as npt


def normalizar(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return (a - a.mean()) / a.std()


print(normalizar([1, 2, 3]).round(3))
print(normalizar(np.array([10, 20, 30])).round(3))


# === Validar a forma em tempo de execução ===

def exigir_forma(a, esperada):
    if a.ndim != len(esperada) or any(e is not None and e != s for e, s in zip(esperada, a.shape)):
        raise ValueError(f"forma {a.shape} incompatível com {esperada}")
    return a


try:
    exigir_forma(np.zeros((3, 4)), (None, 5))
except ValueError as erro:
    print(erro)


# === O que mudou no NumPy 2 ===

removidos = ["float_", "NaN", "Inf", "in1d", "trapz", "round_", "msort"]
print({nome: hasattr(np, nome) for nome in removidos})
print(hasattr(np, "trapezoid"), hasattr(np, "isin"))

print(repr(np.float64(3.0)), str(np.float64(3.0)))

lista = [1, 2, 3]
try:
    np.array(lista, copy=False)
except ValueError as erro:
    print("copy=False falha quando é preciso copiar:", type(erro).__name__)

x = np.array([1.0, 2.0])
print(np.asarray(x, copy=False) is x, np.array(x, copy=None) is x)

print(np.unstack(np.arange(6).reshape(2, 3)))


# === Exercício: Validar uma matriz de entrada ===

def validar_matriz(a):
    m = np.asarray(a, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"esperava 2 dimensões, recebi {m.ndim}")
    if not np.isfinite(m).all():
        raise ValueError("a matriz contém nan ou infinitos")
    return m


assert validar_matriz([[1, 2], [3, 4]]).dtype == np.float64
for ruim in ([1, 2, 3], [[1.0, np.nan]], [[np.inf, 1.0]]):
    try:
        validar_matriz(ruim)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{ruim!r} deveria falhar")
print("ok")
