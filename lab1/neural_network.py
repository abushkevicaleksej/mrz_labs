###############################
# Лабораторная работа №3 по дисциплине МРЗвИС
# Выполнил студент группы 221701 БГУИР Абушкевич Алексей Александрович
# Файл, содержащий реализацию линейной нейронной рециркуляционной сети

import math

import numpy as np

from utils import adaptive_a_func

class NeuralNetwork:
    def __init__(self):
        self.W_f = None
        self.W_b = None

        self.adaptive_calc_func = None

    def init_weights(self, input_length, compress_ratio):
        input_size = input_length
        hidden_size = int(input_size / compress_ratio)

        self.W_f = np.random.uniform(-0.001, 0.001, (input_size, hidden_size)).astype(np.float32)
        self.W_b = np.random.uniform(-0.001, 0.001, (hidden_size, input_size)).astype(np.float32)

    def train(self, X, max_epochs: int, max_error: float):
        epoch = 1

        error = float('inf')

        while (math.isnan(error) or error > max_error) and (epoch <= max_epochs):
            for x in X:
                y = self.compress(x) # Y = X * W_f

                xx = self.decompress(y) # XX = Y * W_b

                diff = xx - x

                adaptive_for_forward = adaptive_a_func(xx)
                adaptive_for_backward = adaptive_a_func(y)

                grad_W_b = np.outer(y, diff)
                grad_W_f = np.outer(x, diff @ self.W_b.T)

                self.W_b -= adaptive_for_backward * grad_W_b
                self.W_f -= adaptive_for_forward * grad_W_f

            source = np.asarray(X)
            decoded = (source @ self.W_f) @ self.W_b

            error = 0.5 * np.sum((decoded - source) ** 2)

            yield epoch, error
            epoch += 1

    def compress(self, X):
        return X @ self.W_f

    def decompress(self, Y):
        return Y @ self.W_b