import argparse

import numpy as np

from rede_neural.dados import dividir, espiral, ou_exclusivo
from rede_neural.rede import RedeDensa
from rede_neural.treino import acuracia, treinar


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rede", description="Rede neural só com NumPy")
    parser.add_argument("--semente", type=int, default=0)
    sub = parser.add_subparsers(dest="problema", required=True)
    espiral_p = sub.add_parser("espiral", help="3 braços de espiral entrelaçados")
    espiral_p.add_argument("--epocas", type=int, default=1000)
    espiral_p.add_argument("--oculta", type=int, default=64, help="neurônios por camada oculta")
    espiral_p.add_argument("--taxa", type=float, default=0.1)
    xor_p = sub.add_parser("xor", help="o ou exclusivo, que nenhuma reta resolve")
    xor_p.add_argument("--epocas", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    rng = np.random.default_rng(args.semente)
    if args.problema == "espiral":
        X, y = espiral(100, 3, 0.2, rng)
        tamanhos = [2, args.oculta, args.oculta, 3]
        taxa = args.taxa
    else:
        X, y = ou_exclusivo(200, rng)
        tamanhos = [2, 8, 2]
        taxa = 0.1
    X_treino, X_teste, y_treino, y_teste = dividir(X, y, 0.2, rng)
    rede = RedeDensa(tamanhos, rng, l2=1e-4)

    passo = max(args.epocas // 5, 1)
    historico = treinar(rede, X_treino, y_treino, epocas=args.epocas, taxa=taxa)
    for epoca in range(passo, args.epocas + 1, passo):
        perda, acc = historico.perdas[epoca - 1], historico.acuracias[epoca - 1]
        print(f"época {epoca:>5}  perda {perda:.4f}  acurácia de treino {acc:.3f}")
    print(f"acurácia no treino: {acuracia(rede, X_treino, y_treino):.3f}")
    print(f"acurácia no teste:  {acuracia(rede, X_teste, y_teste):.3f}")
    return 0
