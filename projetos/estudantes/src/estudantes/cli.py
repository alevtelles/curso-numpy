import argparse
from pathlib import Path

import numpy as np

from estudantes.analise import estudante
from estudantes.dados import gerar_turma
from estudantes.relatorio import montar_relatorio


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="estudantes", description="Análise de uma turma")
    parser.add_argument("--n", type=int, default=200, help="número de estudantes")
    parser.add_argument("--semente", type=int, default=42, help="semente do gerador")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("relatorio", help="mostra o relatório completo")
    aluno = sub.add_parser("aluno", help="mostra os dados de um estudante")
    aluno.add_argument("id", type=int)
    exportar = sub.add_parser("exportar", help="grava a turma em um arquivo CSV")
    exportar.add_argument("arquivo", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    turma = gerar_turma(args.n, args.semente)
    if args.comando == "relatorio":
        print(montar_relatorio(turma))
    elif args.comando == "aluno":
        try:
            dados = estudante(turma, args.id)
        except KeyError as erro:
            print(f"Erro: {erro.args[0]}")
            return 1
        for chave, valor in dados.items():
            print(f"{chave:<14}{valor:g}")
    else:
        tabela = np.column_stack(
            [turma.ids, turma.notas, turma.presenca, turma.horas_estudo, turma.tarefas]
        )
        np.savetxt(
            args.arquivo,
            tabela,
            delimiter=",",
            fmt="%g",
            header="id,nota,presenca,horas_estudo,tarefas",
            comments="",
        )
        print(f"{len(turma)} estudantes gravados em {args.arquivo}")
    return 0
