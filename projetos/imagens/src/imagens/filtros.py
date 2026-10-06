import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from imagens.tipos import Bytes, Reais


def aplicar_kernel(img: Reais, kernel: Reais) -> Reais:
    """Correlação 2D: cada pixel vira a soma ponderada da sua vizinhança.

    As bordas são tratadas repetindo o pixel mais próximo (``mode="edge"``). As janelas são uma
    visão (sem cópia) da imagem, e o ``einsum`` multiplica cada uma pelo kernel de uma vez.
    """
    altura_k, largura_k = kernel.shape
    if altura_k % 2 == 0 or largura_k % 2 == 0:
        raise ValueError("o kernel precisa ter dimensões ímpares")
    margem = ((altura_k // 2, altura_k // 2), (largura_k // 2, largura_k // 2))
    janelas = sliding_window_view(np.pad(img, margem, mode="edge"), (altura_k, largura_k))
    resultado: Reais = np.einsum("ijkl,kl->ij", janelas, kernel)
    return resultado


def kernel_caixa(k: int) -> Reais:
    return np.full((k, k), 1.0 / (k * k))


def kernel_gaussiano(k: int, sigma: float) -> Reais:
    eixo = np.arange(k) - k // 2
    g = np.exp(-(eixo**2) / (2 * sigma**2))
    kernel: Reais = np.outer(g, g)
    return kernel / kernel.sum()


SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def bordas_sobel(img: Bytes) -> Reais:
    """Magnitude do gradiente: grande onde o brilho muda de repente (as bordas)."""
    f = img.astype(np.float64)
    gx = aplicar_kernel(f, SOBEL_X)
    gy = aplicar_kernel(f, SOBEL_X.T)
    return np.hypot(gx, gy)


def para_uint8(img: Reais) -> Bytes:
    """Reescala qualquer array decimal para 0..255, para poder gravar como imagem."""
    menor, maior = float(img.min()), float(img.max())
    if maior == menor:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip((img - menor) / (maior - menor) * 255, 0, 255).round().astype(np.uint8)
