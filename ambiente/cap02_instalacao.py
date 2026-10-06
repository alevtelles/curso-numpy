"""Capítulo 2: Instalando e importando o NumPy.

Parte: Ambiente.
Execute com: python cap02_instalacao.py
"""


# === Importar, e a convenção `np` ===

import sys
from pathlib import Path

import numpy as np

print("versão 2 ou superior:", int(np.__version__.split(".")[0]) >= 2)
print("onde está o NumPy:", Path(np.__file__).parent.name)
print("este Python existe no disco:", Path(sys.executable).exists())

np.show_config()


# === Exercício: Um diagnóstico do ambiente ===

def diagnostico():
    return {
        "numpy": np.__version__,
        "python": sys.version_info[:2],
        "executavel_existe": Path(sys.executable).exists(),
    }


resultado = diagnostico()
assert resultado["executavel_existe"]
assert resultado["python"] >= (3, 12)
assert resultado["numpy"].split(".")[0].isdigit()
print("ok")
