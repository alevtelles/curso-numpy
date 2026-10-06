from pathlib import Path

import numpy as np
import pytest

from imagens.pngio import ASSINATURA, ler_png, salvar_png


def test_ida_e_volta_em_cinza_e_em_rgb(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    cinza = rng.integers(0, 256, (7, 9), dtype=np.uint8)
    rgb = rng.integers(0, 256, (5, 6, 3), dtype=np.uint8)
    salvar_png(tmp_path / "c.png", cinza)
    salvar_png(tmp_path / "r.png", rgb)
    assert np.array_equal(ler_png(tmp_path / "c.png"), cinza)
    assert np.array_equal(ler_png(tmp_path / "r.png"), rgb)


def test_o_arquivo_tem_a_assinatura_do_png(tmp_path: Path) -> None:
    salvar_png(tmp_path / "x.png", np.zeros((2, 2), dtype=np.uint8))
    assert (tmp_path / "x.png").read_bytes().startswith(ASSINATURA)


def test_entradas_invalidas(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uint8"):
        salvar_png(tmp_path / "a.png", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="RGB"):
        salvar_png(tmp_path / "b.png", np.zeros((2, 2, 4), dtype=np.uint8))
    (tmp_path / "texto.png").write_text("não sou uma imagem", encoding="utf-8")
    with pytest.raises(ValueError, match="PNG"):
        ler_png(tmp_path / "texto.png")
