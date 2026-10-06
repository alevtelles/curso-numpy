# %% [markdown]
# # Notas de vendas
# Cada bloco `# %%` é uma célula no VS Code e no JupyterLab (com jupytext).

# %%
import numpy as np

vendas = np.array([120.0, 80.5, 99.9, 150.0])

# %%
print(vendas.mean().round(2))

# %%
print(vendas.max())
