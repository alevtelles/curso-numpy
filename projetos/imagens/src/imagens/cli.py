import argparse
from pathlib import Path

import numpy as np

from imagens.filtros import aplicar_kernel, bordas_sobel, kernel_gaussiano, para_uint8
from imagens.pngio import ler_png, salvar_png
from imagens.sintetica import criar_cena
from imagens.tons import binarizar, equalizar, esticar_contraste, limiar_otsu, para_cinza


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imagens", description="Processamento de imagens")
    sub = parser.add_subparsers(dest="comando", required=True)
    gerar = sub.add_parser("gerar", help="grava uma imagem de teste")
    gerar.add_argument("pasta", type=Path)
    gerar.add_argument("--tamanho", type=int, default=128)
    processar = sub.add_parser("processar", help="aplica uma sequência de operações")
    processar.add_argument("entrada", type=Path)
    processar.add_argument("pasta", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    args.pasta.mkdir(parents=True, exist_ok=True)
    if args.comando == "gerar":
        destino = args.pasta / "cena.png"
        salvar_png(destino, criar_cena(args.tamanho))
        print(f"imagem gravada em {destino}")
        return 0

    try:
        rgb = ler_png(args.entrada)
    except (OSError, ValueError) as erro:
        print(f"Erro: {erro}")
        return 1
    cinza = para_cinza(rgb) if rgb.ndim == 3 else rgb
    suave = aplicar_kernel(cinza.astype(float), kernel_gaussiano(5, 1.2))
    desfocada = np.clip(suave.round(), 0, 255).astype(np.uint8)
    limiar = limiar_otsu(cinza)
    etapas = {
        "cinza": cinza,
        "contraste": esticar_contraste(cinza),
        "equalizada": equalizar(cinza),
        "desfoque": desfocada,
        "bordas": para_uint8(bordas_sobel(desfocada)),
        "binaria": binarizar(cinza, limiar),
    }
    for nome, imagem in etapas.items():
        salvar_png(args.pasta / f"{nome}.png", imagem)
        minimo, maximo, media = int(imagem.min()), int(imagem.max()), imagem.mean()
        print(f"{nome:<11} min={minimo:>3} max={maximo:>3} média={media:6.1f}")
    original = rgb if rgb.ndim == 3 else np.stack([rgb] * 3, axis=-1)
    nomes = ("cinza", "equalizada", "desfoque", "bordas", "binaria")
    tiras = [np.stack([etapas[nome]] * 3, axis=-1) for nome in nomes]
    salvar_png(args.pasta / "montagem.png", np.hstack([original, *tiras]))
    print("montagem.png: original, " + ", ".join(nomes))
    print(f"limiar de Otsu: {limiar}")
    return 0
