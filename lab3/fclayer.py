###############################
# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциальной линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией полносвязного слоя нейросети Джордана-Элмана
# Дата 28.11.2025

from typing import Callable
import numpy as np

class FCLayer: # H
    """Полносвязный слой нейронной сети"""
    
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        # self.biases = np.zeros((output_size, 1))
        self.activation = activation
        self.input = None
        self.output = None
        self.z = None  # Взвешенная сумма до активации
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через слой"""
        self.input = x.reshape(-1, 1)  # Преобразуем в столбец
        self.z = np.dot(self.weights, self.input)
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """Обратный проход через слой с обновлением весов"""
        # Градиент по взвешенной сумме
        dz = self.activation.backward(grad)
        
        # Градиенты по параметрам
        dw = np.dot(dz, self.input.T)
        dx = np.dot(self.weights.T, dz)
        
        # Обновление весов
        self.weights -= learning_rate * dw
        
        return dx