import pytest

from rede_neural.cli import main


def test_xor_pela_linha_de_comando(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["xor", "--epocas", "300"]) == 0
    saida = capsys.readouterr().out
    assert "acurácia no teste" in saida
    assert "época" in saida
