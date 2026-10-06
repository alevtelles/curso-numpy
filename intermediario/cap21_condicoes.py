"""Capítulo 21: Condições vetorizadas: where, select e clip.

Parte: Intermediário.
Execute com: python cap21_condicoes.py
"""


# === `np.where`: o `if` do array ===

import numpy as np

notas = np.array([55, 72, 91, 40, 68])
print(np.where(notas >= 60, "aprovado", "reprovado"))
print(np.where(notas >= 60, notas, 0))
print(np.where(notas > 70))


# === `np.select`: várias faixas ===

condicoes = [notas >= 90, notas >= 70, notas >= 60]
rotulos = ["A", "B", "C"]
print(np.select(condicoes, rotulos, default="D"))


# === Limitar e comparar elemento a elemento ===

print(np.maximum(notas, 50))

x = np.array([-2.0, -0.5, 0.5, 2.0])
print(np.piecewise(x, [x < 0, x >= 0], [lambda t: t ** 2, lambda t: t]))


# === A armadilha: `where` calcula os dois lados ===

den = np.array([2.0, 0.0, 4.0])
with np.errstate(divide="ignore"):
    ingenuo = np.where(den != 0, 1 / den, 0.0)
print(ingenuo)

seguro = np.divide(1.0, den, out=np.zeros_like(den), where=den != 0)
print(seguro)


# === Exercício: Classificar o IMC ===

def classificar_imc(imc):
    condicoes = [imc < 18.5, imc < 25, imc < 30]
    return np.select(condicoes, ["abaixo", "normal", "sobrepeso"], default="obesidade")


resultado = classificar_imc(np.array([17.0, 22.0, 27.5, 33.0]))
assert resultado.tolist() == ["abaixo", "normal", "sobrepeso", "obesidade"]
print("ok")
