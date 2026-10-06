import numpy as np

from rede_neural.tipos import Array, Inteiros


def relu(z: Array) -> Array:
    resultado: Array = np.maximum(z, 0.0)
    return resultado


def softmax(z: Array) -> Array:
    """Probabilidades por linha. Subtrair o máximo evita o overflow do exp (capítulo 36)."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    resultado: Array = e / e.sum(axis=1, keepdims=True)
    return resultado


class RedeDensa:
    """Camadas totalmente conectadas com ReLU e uma saída softmax.

    Convenção de formas: ``X`` é ``(n, entradas)``, o peso da camada i é ``(entradas_i, saidas_i)``
    e a saída é ``(n, classes)``. A forma segue a do scikit-learn: uma linha por exemplo.
    """

    def __init__(self, tamanhos: list[int], rng: np.random.Generator, l2: float = 0.0) -> None:
        if len(tamanhos) < 2:
            raise ValueError("informe ao menos a entrada e a saída")
        self.l2 = l2
        # Inicialização de He: o desvio cresce com o inverso da raiz do número de entradas.
        self.pesos: list[Array] = [
            rng.normal(0.0, np.sqrt(2.0 / entrada), size=(entrada, saida))
            for entrada, saida in zip(tamanhos[:-1], tamanhos[1:], strict=True)
        ]
        self.vieses: list[Array] = [np.zeros(saida) for saida in tamanhos[1:]]

    def parametros(self) -> list[Array]:
        """Os próprios arrays (não cópias): o otimizador e o teste de gradiente os alteram."""
        return [*self.pesos, *self.vieses]

    def _frente(self, X: Array) -> tuple[list[Array], list[Array]]:
        """Devolve a entrada de cada camada e as pré-ativações (antes da ReLU)."""
        entradas = [X]
        pre_ativacoes: list[Array] = []
        a = X
        ultima = len(self.pesos) - 1
        for i, (W, b) in enumerate(zip(self.pesos, self.vieses, strict=True)):
            z = a @ W + b
            pre_ativacoes.append(z)
            a = z if i == ultima else relu(z)
            entradas.append(a)
        return entradas, pre_ativacoes

    def probabilidades(self, X: Array) -> Array:
        _, pre_ativacoes = self._frente(X)
        return softmax(pre_ativacoes[-1])

    def prever(self, X: Array) -> Inteiros:
        previsto: Inteiros = self.probabilidades(X).argmax(axis=1)
        return previsto

    def perda_e_gradientes(self, X: Array, y: Inteiros) -> tuple[float, list[Array]]:
        """Perda de entropia cruzada (mais L2) e o gradiente em relação a cada parâmetro."""
        n = len(X)
        entradas, pre_ativacoes = self._frente(X)
        probs = softmax(pre_ativacoes[-1])
        perda = -np.log(np.maximum(probs[np.arange(n), y], 1e-12)).mean()
        perda += 0.5 * self.l2 * sum(float((W**2).sum()) for W in self.pesos)

        # O gradiente da entropia cruzada com softmax é simplesmente (probabilidade - rótulo) / n.
        dz = probs.copy()
        dz[np.arange(n), y] -= 1.0
        dz /= n

        grad_pesos: list[Array] = []
        grad_vieses: list[Array] = []
        for i in range(len(self.pesos) - 1, -1, -1):
            grad_pesos.insert(0, entradas[i].T @ dz + self.l2 * self.pesos[i])
            grad_vieses.insert(0, dz.sum(axis=0))
            if i > 0:
                dz = (dz @ self.pesos[i].T) * (pre_ativacoes[i - 1] > 0)
        return float(perda), [*grad_pesos, *grad_vieses]
