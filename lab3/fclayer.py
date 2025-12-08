# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией полносвязного слоя нейросети Джордана-Элмана
# Дата 28.11.2025

from typing import Callable
import numpy as np

class FCLayer:
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=(output_size, input_size)).astype(np.float32)
        self.activation = activation
        self.input = None
        self.output = None
        self.z = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x.reshape(-1, 1)
        self.z = np.dot(self.weights, self.input)
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        dz = self.activation.backward(grad)
        
        dw = np.dot(dz, self.input.T)
        dx = np.dot(self.weights.T, dz)
        
        self.weights -= learning_rate * dw
        
        return dx