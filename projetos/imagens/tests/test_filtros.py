import numpy as np
import pytest

from imagens.filtros import (
    aplicar_kernel,
    bordas_sobel,
    kernel_caixa,
    kernel_gaussiano,
    para_uint8,
)


def aplicar_por_laco(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Versão lenta e obviamente correta, usada para validar a vetorizada."""
    k = kernel.shape[0] // 2
    padded = np.pad(img, k, mode="edge")
    saida = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            saida[i, j] = (padded[i : i + 2 * k + 1, j : j + 2 * k + 1] * kernel).sum()
    return saida


def test_confere_com_a_versao_por_laco() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((6, 7))
    kernel = rng.random((3, 3))
    assert np.allclose(aplicar_kernel(img, kernel), aplicar_por_laco(img, kernel))


def test_kernel_de_tamanho_par_e_recusado() -> None:
    with pytest.raises(ValueError, match="ímpares"):
        aplicar_kernel(np.zeros((4, 4)), np.ones((2, 2)))


def test_desfoque_preserva_imagem_constante_e_reduz_ruido() -> None:
    assert np.allclose(aplicar_kernel(np.full((5, 5), 7.0), kernel_caixa(3)), 7.0)
    ruido = np.random.default_rng(1).normal(0, 10, (40, 40))
    assert aplicar_kernel(ruido, kernel_gaussiano(5, 1.2)).std() < 0.5 * ruido.std()


def test_kernel_gaussiano_soma_1_e_e_simetrico() -> None:
    k = kernel_gaussiano(5, 1.0)
    assert np.isclose(k.sum(), 1.0)
    assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])


def test_sobel_acha_a_borda_vertical() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    img[:, 5:] = 255
    mag = bordas_sobel(img)
    assert set(np.argmax(mag, axis=1)) <= {4, 5}
    assert np.allclose(mag[:, :3], 0) and np.allclose(mag[:, 7:], 0)


def test_para_uint8_reescala_para_0_255() -> None:
    r = para_uint8(np.array([[-5.0, 0.0], [5.0, 10.0]]))
    assert r.dtype == np.uint8 and r.min() == 0 and r.max() == 255
    assert para_uint8(np.ones((2, 2))).max() == 0
