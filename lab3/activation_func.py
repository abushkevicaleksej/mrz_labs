###############################
# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциальной линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией функций активаций
# Дата 28.11.2025

import numpy as np

class ELU:
    """Экспоненциальная линейная функция активации (ELU)"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через функцию активации"""
        self.input = x.copy()
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Обратный проход через функцию активации"""
        if self.input is None:
            raise ValueError("Необходимо сначала выполнить прямой проход")
        
        derivative = np.where(self.input > 0, 1, self.alpha * np.exp(self.input))
        return grad * derivative

class Linear:
    """Линейная функция активации (для эффекторного слоя)"""
    
    def __init__(self):
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через функцию активации"""
        self.input = x.copy()
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Обратный проход через функцию активации"""
        return grad

class Tanh:
    """Гиперболический тангенс в качестве функции активации"""
    
    def __init__(self):
        self.output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через функцию активации"""
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Обратный проход через функцию активации"""
        if self.output is None:
            raise ValueError("Необходимо сначала выполнить прямой проход")
        
        return grad * (1 - self.output ** 2)