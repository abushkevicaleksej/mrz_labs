import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        n_patterns = patterns.shape[0]
        self.W = np.zeros((self.n_neurons, self.n_neurons))

        for p in patterns:
            self.W += np.outer(p, p)

        self.W /= self.n_neurons
        np.fill_diagonal(self.W, 0)

        # np.savetxt("matrix_W.txt", self.W, delimiter="  ", encoding='utf-8')

    def predict(self, input_pattern, max_iter=1000):
        S = np.copy(input_pattern)
        S[S == 0] = 1

        prev_S = np.zeros_like(S)
        iter_count = 0

        indices = list(range(self.n_neurons))

        for iteration in range(max_iter):
            iter_count = iteration + 1

            print(f'Итерация {iteration}')

            prev_S[:] = S

            np.random.shuffle(indices)
            for i in indices:
                h_i = np.dot(self.W[i, :], S)
                S[i] = 1 if h_i >= 0 else -1

            if np.array_equal(S, prev_S):
                print(f"Достигнуто устойчивое состояние на итерации {iter_count}")
                break

        if iter_count >= max_iter:
            print(f"Достигнуто максимальное количество итераций: {max_iter}")

        return S, iter_count

    def check_stability(self, pattern):
        reconstructed, _ = self.predict(pattern)
        return np.array_equal(pattern, reconstructed)