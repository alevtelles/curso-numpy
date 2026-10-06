import numpy as np

from imagens.tipos import Bytes


def criar_cena(tamanho: int = 128, semente: int = 7) -> Bytes:
    """Imagem RGB de teste: fundo em gradiente, um círculo claro, um retângulo escuro e ruído."""
    rng = np.random.default_rng(semente)
    y, x = np.mgrid[0:tamanho, 0:tamanho]
    cena = np.zeros((tamanho, tamanho, 3), dtype=np.float64)
    cena[..., 0] = 60 + 80 * x / (tamanho - 1)
    cena[..., 1] = 60 + 80 * y / (tamanho - 1)
    cena[..., 2] = 90

    centro = tamanho * 0.35
    circulo = (x - centro) ** 2 + (y - centro) ** 2 <= (tamanho * 0.2) ** 2
    cena[circulo] = (230, 200, 60)
    retangulo = (x > tamanho * 0.55) & (x < tamanho * 0.9)
    retangulo &= (y > tamanho * 0.5) & (y < tamanho * 0.85)
    cena[retangulo] = (40, 60, 200)

    cena += rng.normal(0, 6, cena.shape)
    return np.clip(cena, 0, 255).round().astype(np.uint8)
