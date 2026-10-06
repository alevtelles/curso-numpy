from pathlib import Path

import pytest

from imagens.cli import main


def test_gerar_e_processar_de_ponta_a_ponta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["gerar", str(tmp_path), "--tamanho", "64"]) == 0
    assert main(["processar", str(tmp_path / "cena.png"), str(tmp_path / "saida")]) == 0
    saida = capsys.readouterr().out
    assert "limiar de Otsu" in saida
    nomes = {p.name for p in (tmp_path / "saida").glob("*.png")}
    assert nomes == {
        "cinza.png",
        "contraste.png",
        "equalizada.png",
        "desfoque.png",
        "bordas.png",
        "binaria.png",
        "montagem.png",
    }


def test_arquivo_que_nao_e_png_devolve_codigo_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    falso = tmp_path / "falso.png"
    falso.write_text("nada", encoding="utf-8")
    assert main(["processar", str(falso), str(tmp_path / "s")]) == 1
    assert "Erro" in capsys.readouterr().out
