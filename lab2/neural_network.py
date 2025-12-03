###############################
# Лабораторная работа №4 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Хопфилда с дискретным состоянием и дискретным временем в асинхронном режиме.
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл, содержащий реализацию сети Хопфилда
# Дата 17.11.2025

import numpy as np

def activation_func(prev_S, h_i):
    if h_i > 0:
        return 1
    elif h_i < 0:
        return -1
    else:
        return prev_S

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        for x in patterns:
            x = x.reshape(-1, 1)  

            Wx = self.W @ x
            diff = Wx - x         
            numerator = diff @ diff.T

            denom = (x.T @ x) - (x.T @ Wx)
            denom = denom.item()

            if denom == 0:
                continue

            self.W = self.W + numerator / denom

        np.fill_diagonal(self.W, 0)

        # np.savetxt("matrix_W.txt", self.W, delimiter="  ", encoding='utf-8')

    def predict(self, input_pattern, max_iter=1000):
        S = np.copy(input_pattern)
        S[S == 0] = 1

        prev_S = np.zeros_like(S)
        iter_count = 0

        energy_history = []

        initial_energy = self.calculate_energy(S)
        energy_history.append(initial_energy)

        print(f"Начальная энергия: {initial_energy:.4f}")

        indices = list(range(self.n_neurons))

        for iteration in range(max_iter):
            iter_count = iteration + 1

            prev_S[:] = S

            np.random.shuffle(indices)
            for i in indices:
                h_i = np.dot(self.W[i, :], S)
                S[i] = activation_func(prev_S[i], h_i)

            current_energy = self.calculate_energy(S)
            energy_history.append(current_energy)

            # energy_change = current_energy - energy_history[-2] if iter_count > 1 else 0
            print(f"Итерация {iter_count}, энергия {current_energy:.4f}")

            if np.array_equal(S, prev_S):
                print(f"Достигнуто устойчивое состояние на итерации {iter_count}")
                break

        if iter_count >= max_iter:
            print(f"Достигнуто максимальное количество итераций: {max_iter}")

        return S, iter_count

    def calculate_energy(self, state):
        energy = 0
        for i in range(self.n_neurons):
            for j in range(i + 1, self.n_neurons):
                energy += self.W[i, j] * state[i] * state[j]
        return -energy

    def check_stability(self, pattern):
        reconstructed, _ = self.predict(pattern)
        return np.array_equal(pattern, reconstructed)