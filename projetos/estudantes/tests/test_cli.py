from pathlib import Path

import numpy as np
import pytest

from estudantes.cli import main


def test_relatorio(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["relatorio"]) == 0
    saida = capsys.readouterr().out
    assert "Turma com 200 estudantes" in saida
    assert "Nível 3" in saida


def test_aluno_inexistente_devolve_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["aluno", "9999"]) == 1
    assert "não existe o estudante 9999" in capsys.readouterr().out


def test_exportar_csv(tmp_path: Path) -> None:
    destino = tmp_path / "turma.csv"
    assert main(["--n", "30", "exportar", str(destino)]) == 0
    lido = np.loadtxt(destino, delimiter=",", skiprows=1)
    assert lido.shape == (30, 5)
