import numpy as np
import numpy.typing as npt

from imagens.tipos import Bytes

PESOS_CINZA = np.array([0.299, 0.587, 0.114])


def para_cinza(rgb: Bytes) -> Bytes:
    """Média ponderada dos canais, com os pesos da percepção humana (verde pesa mais)."""
    cinza: Bytes = np.clip((rgb[..., :3] @ PESOS_CINZA).round(), 0, 255).astype(np.uint8)
    return cinza


def esticar_contraste(img: Bytes) -> Bytes:
    """Leva o menor valor a 0 e o maior a 255, mantendo a proporção entre os demais."""
    menor, maior = int(img.min()), int(img.max())
    if maior == menor:
        return img.copy()
    esticada = (img.astype(np.float64) - menor) * (255 / (maior - menor))
    return np.clip(esticada.round(), 0, 255).astype(np.uint8)


def histograma(img: Bytes) -> npt.NDArray[np.intp]:
    return np.bincount(img.ravel(), minlength=256)


def equalizar(img: Bytes) -> Bytes:
    """Redistribui os níveis para o histograma ficar o mais uniforme possível."""
    cdf = histograma(img).cumsum()
    cdf_min = int(cdf[np.flatnonzero(histograma(img))[0]])
    if cdf[-1] == cdf_min:
        return img.copy()
    tabela = np.clip(np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255), 0, 255).astype(np.uint8)
    equalizada: Bytes = tabela[img]
    return equalizada


def limiar_otsu(img: Bytes) -> int:
    """Limiar que separa a imagem em dois grupos com a maior variância entre eles (Otsu)."""
    prob = histograma(img) / img.size
    niveis = np.arange(256)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * niveis)
    with np.errstate(divide="ignore", invalid="ignore"):
        variancia_entre = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.argmax(np.nan_to_num(variancia_entre)))


def binarizar(img: Bytes, limiar: int) -> Bytes:
    return np.where(img > limiar, 255, 0).astype(np.uint8)
