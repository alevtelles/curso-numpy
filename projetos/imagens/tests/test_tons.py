import numpy as np

from imagens.tons import (
    binarizar,
    equalizar,
    esticar_contraste,
    histograma,
    limiar_otsu,
    para_cinza,
)


def test_cinza_de_cores_conhecidas() -> None:
    vermelho = np.array([[[255, 0, 0]]], dtype=np.uint8)
    branco = np.array([[[255, 255, 255]]], dtype=np.uint8)
    assert para_cinza(vermelho)[0, 0] == 76
    assert para_cinza(branco)[0, 0] == 255


def test_esticar_contraste_leva_aos_extremos() -> None:
    img = np.array([[100, 120], [140, 150]], dtype=np.uint8)
    r = esticar_contraste(img)
    assert r.min() == 0 and r.max() == 255
    assert np.array_equal(esticar_contraste(np.full((2, 2), 9, dtype=np.uint8)), np.full((2, 2), 9))


def test_histograma_soma_o_numero_de_pixels() -> None:
    img = np.random.default_rng(0).integers(0, 256, (10, 10), dtype=np.uint8)
    assert histograma(img).sum() == img.size


def test_equalizar_aumenta_a_dispersao_de_uma_imagem_de_baixo_contraste() -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(100, 140, (40, 40), dtype=np.uint8)
    r = equalizar(img)
    assert r.std() > 2 * img.std()
    assert r.min() == 0 and r.max() == 255


def test_otsu_separa_duas_populacoes() -> None:
    rng = np.random.default_rng(3)
    claro = np.clip(rng.normal(200, 10, 500), 0, 255)
    escuro = np.clip(rng.normal(50, 10, 500), 0, 255)
    img = np.concatenate([claro, escuro]).astype(np.uint8).reshape(25, 40)
    assert 80 < limiar_otsu(img) < 170
    assert set(np.unique(binarizar(img, limiar_otsu(img)))) == {0, 255}
