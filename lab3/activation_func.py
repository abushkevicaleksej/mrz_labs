import numpy as np
import random
random.seed(42)
np.random.seed(42)

class ELU:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.input = None # Здесь храним Z (до активации)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x.copy()
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Производная ELU: 1, если x > 0, иначе alpha * exp(x)
        # Важно: используем self.input (значение до активации), а не выход
        derivative = np.where(self.input > 0, 1.0, self.alpha * np.exp(self.input))
        return grad * derivative

class Tanh:
    def __init__(self):
        self.output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - self.output ** 2)

class Linear:
    def __init__(self):
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad