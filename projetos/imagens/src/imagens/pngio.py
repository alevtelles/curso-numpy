"""Leitura e gravação de PNG com a biblioteca padrão e NumPy.

Grava imagens de 8 bits em tons de cinza ou RGB, sem filtro por linha (filtro 0). Lê apenas
esse mesmo formato, ou seja, os arquivos gravados por este módulo.
"""

import struct
import zlib
from pathlib import Path

import numpy as np

from imagens.tipos import Bytes

ASSINATURA = b"\x89PNG\r\n\x1a\n"


def _bloco(tipo: bytes, dados: bytes) -> bytes:
    return (
        struct.pack(">I", len(dados)) + tipo + dados + struct.pack(">I", zlib.crc32(tipo + dados))
    )


def salvar_png(caminho: Path, img: Bytes) -> None:
    if img.dtype != np.uint8:
        raise ValueError("a imagem precisa ser uint8")
    if img.ndim == 2:
        tipo_cor = 0
    elif img.ndim == 3 and img.shape[2] == 3:
        tipo_cor = 2
    else:
        raise ValueError("use uma imagem 2D (cinza) ou (altura, largura, 3) (RGB)")
    altura, largura = img.shape[:2]
    # Cada linha começa com um byte de filtro (0 = sem filtro).
    linhas = np.concatenate(
        [np.zeros((altura, 1), dtype=np.uint8), img.reshape(altura, -1)], axis=1
    )
    cabecalho = struct.pack(">IIBBBBB", largura, altura, 8, tipo_cor, 0, 0, 0)
    conteudo = (
        ASSINATURA
        + _bloco(b"IHDR", cabecalho)
        + _bloco(b"IDAT", zlib.compress(linhas.tobytes(), 9))
        + _bloco(b"IEND", b"")
    )
    caminho.write_bytes(conteudo)


def ler_png(caminho: Path) -> Bytes:
    dados = caminho.read_bytes()
    if not dados.startswith(ASSINATURA):
        raise ValueError("não é um arquivo PNG")
    posicao = len(ASSINATURA)
    largura = altura = tipo_cor = 0
    comprimidos = b""
    while posicao < len(dados):
        (tamanho,) = struct.unpack(">I", dados[posicao : posicao + 4])
        tipo = dados[posicao + 4 : posicao + 8]
        corpo = dados[posicao + 8 : posicao + 8 + tamanho]
        if tipo == b"IHDR":
            largura, altura, profundidade, tipo_cor, *_ = struct.unpack(">IIBBBBB", corpo)
            if profundidade != 8 or tipo_cor not in (0, 2):
                raise ValueError("só leio PNG de 8 bits em cinza ou RGB")
        elif tipo == b"IDAT":
            comprimidos += corpo
        posicao += 12 + tamanho
    canais = 1 if tipo_cor == 0 else 3
    bruto = np.frombuffer(zlib.decompress(comprimidos), dtype=np.uint8)
    linhas = bruto.reshape(altura, 1 + largura * canais)
    if linhas[:, 0].any():
        raise ValueError("só leio PNG gravado sem filtro por linha (como o deste módulo)")
    pixels = linhas[:, 1:]
    return pixels.reshape(altura, largura) if canais == 1 else pixels.reshape(altura, largura, 3)
