"""Capítulo 3: Notebooks, scripts e o fluxo de trabalho.

Parte: Ambiente.
Execute com: python cap03_notebooks.py
"""


# === O kernel e a armadilha do estado escondido ===

celulas = ["total = 10", "dobro = total * 2"]

kernel = {}
for celula in celulas:
    exec(celula, kernel)

celulas[0] = "total = 99"
exec(celulas[0], kernel)
print("o que a tela mostra:", kernel["dobro"])

novo_kernel = {}
for celula in celulas:
    exec(celula, novo_kernel)
print("depois de reiniciar e rodar tudo:", novo_kernel["dobro"])
