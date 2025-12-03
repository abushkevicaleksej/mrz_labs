from typing import Callable
import numpy as np

class FCLayer:
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        # ИСПРАВЛЕНИЕ 1: Правильная инициализация весов (He Initialization)
        # Это критически важно. Веса будут порядка ~0.3-0.5, а не 0.01
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(output_size, input_size) * scale
        
        # ИСПРАВЛЕНИЕ 2: Добавление смещения (Bias)
        self.bias = np.zeros((output_size, 1))
        
        self.activation = activation
        self.input = None
        self.output = None
        self.z = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x.reshape(-1, 1)
        # Z = Wx + b
        self.z = np.dot(self.weights, self.input) + self.bias
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        # Градиент по выходу функции активации
        dz = self.activation.backward(grad)
        
        # Градиенты по весам, смещению и входу
        dw = np.dot(dz, self.input.T)
        db = np.sum(dz, axis=1, keepdims=True) # Градиент для bias
        dx = np.dot(self.weights.T, dz)
        
        # Обновление параметров
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return dx
        
    # Метод plot_uniform_weights можно удалить или оставить старым, 
    # но он больше не отражает реальное распределение (теперь оно нормальное, а не uniform)