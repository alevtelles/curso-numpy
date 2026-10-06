"""Capítulo 1: Por que Python virou a linguagem dos dados.

Parte: Ambiente.
Execute com: python cap01_ecossistema_python.py
"""


# === Python é pequeno, e o ecossistema é grande ===

import importlib.util
import sys

import numpy as np

print("numpy faz parte da biblioteca padrão?", "numpy" in sys.stdlib_module_names)
print("numpy está instalado aqui?", importlib.util.find_spec("numpy") is not None)
print(type(np.add).__name__, type(np.ndarray).__name__)


# === Exercício: Quais destes módulos vêm com o Python? ===

modulos = ["math", "numpy", "json", "pandas"]
da_biblioteca_padrao = [m for m in modulos if m in sys.stdlib_module_names]
assert da_biblioteca_padrao == ["math", "json"]
print("ok")
