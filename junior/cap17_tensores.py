"""Capítulo 17: Tensores: do escalar ao lote de imagens.

Parte: Básico.
Execute com: python cap17_tensores.py
"""


# === As dimensões, de zero a cinco ===

import numpy as np

escalar = np.array(7)
vetor = np.array([255, 0, 0])
matriz = np.zeros((28, 28))
imagem = np.zeros((224, 224, 3), dtype=np.uint8)
lote = np.zeros((32, 224, 224, 3), dtype=np.float32)

for nome, t in [("escalar", escalar), ("vetor", vetor), ("matriz", matriz), ("imagem", imagem), ("lote", lote)]:
    print(f"{nome:<8} ndim={t.ndim} shape={t.shape}")
print(round(lote.nbytes / 1e6, 1), "MB")


# === Ler uma forma como um idioma ===

chw = np.moveaxis(imagem, -1, 0)
print(chw.shape, np.transpose(chw, (1, 2, 0)).shape)


# === A mesma regra de sempre, em mais dimensões ===

print(lote.mean(axis=(0, 1, 2)).shape)


# === Uma imagem de verdade, em miniatura ===

pixels = np.array(
    [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
    dtype=np.uint8,
)
print(pixels.shape)
print(pixels[0, 1])
print(pixels[..., 0])
cinza = pixels.mean(axis=-1).astype(np.uint8)
print(cinza)


# === Exercício: Empilhar imagens em um lote ===

def para_lote(imagens):
    return np.stack(imagens)


imagens = [np.zeros((8, 8, 3)) for _ in range(4)]
assert para_lote(imagens).shape == (4, 8, 8, 3)
print("ok")
