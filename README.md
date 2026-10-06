# Curso Numpy: código dos capítulos

![Imagem do Curso Numpy](Numpy.png)

Repositório de código do livro **Curso Numpy**: do primeiro array à rede neural, em português.
Cada capítulo tem um arquivo `.py` executável, na mesma ordem em que o código aparece no livro.
O livro em HTML está em `docs/numpy-notes.html`.

## Requisitos

- Python 3.12 ou superior (os arquivos foram executados com Python 3.12.3)
- NumPy 2.x (os arquivos foram executados com a versão 2.4.4)
- Os capítulos do Ecossistema usam pandas, matplotlib e scikit-learn. Instale-os com `uv sync --group ecossistema`
- `pytest` para os projetos

## Como rodar

Cada arquivo roda sozinho, a partir da raiz do repositório:

```bash
python3 junior/cap05_criando_arrays.py
```

No Windows, use `py` no lugar de `python3`. Com uv, sem instalar nada antes:

```bash
uv run junior/cap05_criando_arrays.py
```

Alguns capítulos gravam arquivos (`.npy`, `.npz`, `.csv`) na pasta onde você executa o comando.
Esses arquivos estão no `.gitignore`.

Para os testes de um projeto:

```bash
cd projetos/estudantes
uv sync
uv run pytest
```

## Capítulos

### Ambiente

| Cap. | Capítulo | Arquivo |
|---|---|---|
| 1 | Por que Python virou a linguagem dos dados | `ambiente/cap01_ecossistema_python.py` |
| 2 | Instalando e importando o NumPy | `ambiente/cap02_instalacao.py` |
| 3 | Notebooks, scripts e o fluxo de trabalho | `ambiente/cap03_notebooks.py` |
| 4 | Por que o NumPy é rápido | `ambiente/cap04_por_que_rapido.py` |
### Básico

| Cap. | Capítulo | Arquivo |
|---|---|---|
| 5 | Criando arrays | `junior/cap05_criando_arrays.py` |
| 6 | Anatomia de um array | `junior/cap06_anatomia.py` |
| 7 | Tipos de dados: dtype | `junior/cap07_tipos_dtype.py` |
| 8 | Indexação e fatiamento | `junior/cap08_indexacao.py` |
| 9 | Indexação booleana e fancy | `junior/cap09_indexacao_booleana.py` |
| 10 | Cópias e views | `junior/cap10_copias_views.py` |
| 11 | Reshape e transposição | `junior/cap11_reshape.py` |
| 12 | Vetorização | `junior/cap12_vetorizacao.py` |
| 13 | Broadcasting | `junior/cap13_broadcasting.py` |
| 14 | Agregações e o parâmetro axis | `junior/cap14_agregacoes.py` |
| 15 | Funções universais (ufuncs) | `junior/cap15_ufuncs.py` |
| 16 | Números aleatórios com Generator | `junior/cap16_aleatorios.py` |
| 17 | Tensores: do escalar ao lote de imagens | `junior/cap17_tensores.py` |
| 18 | Medir desempenho de verdade | `junior/cap18_medir_desempenho.py` |
### Intermediário

| Cap. | Capítulo | Arquivo |
|---|---|---|
| 19 | Juntar, empilhar e dividir | `intermediario/cap19_juntar_dividir.py` |
| 20 | Ordenar, buscar e contar | `intermediario/cap20_ordenar_buscar.py` |
| 21 | Condições vetorizadas: where, select e clip | `intermediario/cap21_condicoes.py` |
| 22 | Estatística: percentis, histogramas e correlação | `intermediario/cap22_estatistica.py` |
| 23 | Álgebra linear | `intermediario/cap23_algebra_linear.py` |
| 24 | Ler e gravar arrays | `intermediario/cap24_ler_gravar.py` |
| 25 | Dados ausentes: NaN e arrays mascarados | `intermediario/cap25_dados_ausentes.py` |
| 26 | Tipos estruturados e datas | `intermediario/cap26_estruturados_datas.py` |
| 27 | Indexação avançada e malhas | `intermediario/cap27_indexacao_avancada.py` |
| 28 | Janelas deslizantes e séries | `intermediario/cap28_janelas_series.py` |
### Avançado

| Cap. | Capítulo | Arquivo |
|---|---|---|
| 29 | Memória por dentro: strides, contiguidade e ordem | `avancado/cap29_memoria_strides.py` |
| 30 | einsum e tensordot | `avancado/cap30_einsum.py` |
| 31 | Vetorizar problemas difíceis | `avancado/cap31_vetorizar_dificil.py` |
| 32 | Desempenho: temporários, in-place e dtypes | `avancado/cap32_desempenho_numpy.py` |
| 33 | Arquivos maiores que a memória: memmap | `avancado/cap33_memmap.py` |
| 34 | Ufuncs por dentro | `avancado/cap34_ufuncs_por_dentro.py` |
| 35 | Aleatoriedade avançada | `avancado/cap35_aleatoriedade_avancada.py` |
| 36 | Testes numéricos e estabilidade | `avancado/cap36_testes_numericos.py` |
| 37 | Tipagem, API e o NumPy 2 | `avancado/cap37_tipagem_numpy2.py` |
### Ecossistema

| Cap. | Capítulo | Arquivo |
|---|---|---|
| 38 | NumPy e pandas | `ecossistema/cap38_numpy_pandas.py` |
| 39 | NumPy e Matplotlib | `ecossistema/cap39_numpy_matplotlib.py` |
| 40 | Do NumPy ao aprendizado de máquina | `ecossistema/cap40_numpy_sklearn.py` |
### Projetos

| Cap. | Capítulo | Arquivo |
|---|---|---|
| 41 | Projeto Básico: desempenho de estudantes | (sem arquivo .py) |
| 42 | Projeto Intermediário: imagens só com NumPy | (sem arquivo .py) |
| 43 | Projeto Avançado: uma rede neural do zero | (sem arquivo .py) |

## Convenções

- Um arquivo por capítulo, nomeado `capNN_assunto.py`.
- Os comentários `# === Seção ===` repetem os títulos do livro.
- Arquivos de apoio dos capítulos ficam em `exemplos/`.
- Os três projetos integradores ficam em `projetos/`, cada um com os seus próprios testes.

## Autor

**Alexsander Valente**, Software & AI Architecture, Product Engineering e Data Engineering.

- Site: https://www.alexsander.app
- LinkedIn: https://www.linkedin.com/in/alexsander-valente/
