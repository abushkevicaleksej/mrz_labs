###############################
# Лабораторная работа №3 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами.
# Выполнил студент группы 221701 БГУИР Абушкевич Алексей Александрович
# Файл, содержащий реализацию линейной нейронной рециркуляционной сети
# Источники: Головко В. А . Нейроинтеллект: теория и применение. Книга 1: Организация и обучение нейронных сетей с прямым и обратными связями. Брест Изд. БПИ, 1999-264 с.
# Дата 25.10.2025

import math

import numpy as np

from utils import adaptive_a_func, check_normality, truncated_normal

class NeuralNetwork:
    def __init__(self):
        self.W_f = None
        self.W_b = None

        self.adaptive_calc_func = None

    def init_weights(self, input_length, p):
        input_size = input_length
        hidden_size = p

        # self.W_f = truncated_normal((input_size, hidden_size), std=0.001/3)
        # self.W_b = truncated_normal((hidden_size, input_size), std=0.001/3)

        self.W_f = np.random.uniform(low=-0.01, high=0.01, size=(input_size, hidden_size)).astype(np.float32)
        self.W_b = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, input_size)).astype(np.float32)

        # print(f'Проверка нормальности распределения для W_f: {check_normality(self.W_f)}')
        # print(f'Проверка нормальности распределения для W_b: {check_normality(self.W_b)}')

    def train(self, X, max_epochs: int, max_error: float):
        epoch = 1

        error = float('inf')

        source = np.asarray(X)
        decoded = (source @ self.W_f) @ self.W_b

        error = 0.5 * np.sum((decoded - source) ** 2)

        print(f"до {error}")
        while (math.isnan(error) or error > max_error) and (epoch <= max_epochs):
            for x in X:
                y = self.compress(x) # Y = X * W_f

                xx = self.decompress(y) # XX = Y * W_b

                diff = xx - x

                adaptive_for_forward = adaptive_a_func(x)
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