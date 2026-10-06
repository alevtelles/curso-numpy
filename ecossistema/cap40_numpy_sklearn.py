"""Capítulo 40: Do NumPy ao aprendizado de máquina.

Parte: Ecossistema.
Execute com: python cap40_numpy_sklearn.py
"""


# === A mesma regressão, de dois jeitos ===

import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
verdadeiro = np.array([2.0, -1.0, 0.5])
y = X @ verdadeiro + 3.0 + rng.normal(0, 0.1, 200)

Xb = np.hstack([X, np.ones((200, 1))])
theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)

modelo = LinearRegression().fit(X, y)
print(np.allclose(theta[:3], modelo.coef_), np.isclose(theta[3], modelo.intercept_))
print(np.allclose(theta[:3], verdadeiro, atol=0.05))


# === Medir o erro à mão ===

def r2(real, previsto):
    return 1 - ((real - previsto) ** 2).sum() / ((real - real.mean()) ** 2).sum()


print(round(r2(y, modelo.predict(X)), 3) == round(modelo.score(X, y), 3))


# === Avaliar em dados que o modelo não viu ===

ordem = rng.permutation(len(y))
treino, teste = np.split(ordem, [150])
Xb_treino = np.hstack([X[treino], np.ones((150, 1))])
theta_t, *_ = np.linalg.lstsq(Xb_treino, y[treino], rcond=None)
previsto = np.hstack([X[teste], np.ones((50, 1))]) @ theta_t
print(r2(y[teste], previsto) > 0.99)


# === Descida do gradiente: aprender sem a fórmula fechada ===

w = np.zeros(4)
for _ in range(500):
    w -= 0.1 * (2 / len(y)) * Xb.T @ (Xb @ w - y)
print(np.allclose(w, theta, atol=1e-3))


# === Padronização: o que o `StandardScaler` faz ===

from sklearn.preprocessing import StandardScaler

print(np.allclose(StandardScaler().fit_transform(X), (X - X.mean(axis=0)) / X.std(axis=0)))


# === Exercício: Um classificador do centroide mais próximo ===

from sklearn.neighbors import NearestCentroid


def classificador_centroide(X_treino, y_treino, X_teste):
    classes = np.unique(y_treino)
    centros = np.array([X_treino[y_treino == c].mean(axis=0) for c in classes])
    distancias = np.linalg.norm(X_teste[:, None, :] - centros[None, :, :], axis=-1)
    return classes[distancias.argmin(axis=1)]


gerador = np.random.default_rng(1)
X0 = gerador.normal([0, 0], 1.0, size=(60, 2))
X1 = gerador.normal([4, 4], 1.0, size=(60, 2))
Xc = np.vstack([X0, X1])
yc = np.array([0] * 60 + [1] * 60)
Xt = gerador.normal([2, 2], 2.5, size=(40, 2))

minha = classificador_centroide(Xc, yc, Xt)
deles = NearestCentroid().fit(Xc, yc).predict(Xt)
assert np.array_equal(minha, deles)
print("ok")
