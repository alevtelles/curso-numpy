from estudantes.analise import (
    correlacoes,
    melhores,
    nivel_1,
    nivel_2,
    nivel_3,
    perfil_do_grupo,
)
from estudantes.dados import Turma


def montar_relatorio(t: Turma) -> str:
    linhas = [f"Turma com {len(t)} estudantes", "", "Nível 1: estatísticas das notas"]
    for nome, valor in nivel_1(t).items():
        linhas.append(f"  {nome:<14}{valor:>8.2f}")

    linhas += ["", "Nível 2: grupos por uma condição"]
    for nome, ids in nivel_2(t).items():
        linhas.append(f"  {nome:<28}{len(ids):>4} estudantes")

    linhas += ["", "Nível 3: grupos por várias condições"]
    grupos = nivel_3(t)
    for nome, ids in grupos.items():
        linhas.append(f"  {nome:<28}{len(ids):>4} estudantes")

    linhas += ["", "Correlação de cada variável com a nota"]
    for nome, valor in correlacoes(t).items():
        linhas.append(f"  {nome:<14}{valor:>7.2f}")

    perfil = perfil_do_grupo(t, grupos["presenca_alta_nota_baixa"])
    if perfil:
        linhas += ["", "Perfil de quem tem presença alta e nota baixa"]
        linhas.append(f"  horas de estudo médias {perfil['horas_estudo_medias']:.1f}")
        linhas.append(f"  tarefas médias         {perfil['tarefas_medias']:.1f}")

    linhas += ["", "Os 5 melhores (ids): " + ", ".join(str(int(i)) for i in melhores(t, 5))]
    return "\n".join(linhas)
