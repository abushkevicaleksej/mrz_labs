from typing import Callable
import numpy as np
import random
random.seed(42)
np.random.seed(42)

class FCLayer:
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(output_size, input_size)).astype(np.float32)
        
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
        dz = self.activation.backward(grad)
        
        dw = np.dot(dz, self.input.T)
        db = dz  # Градиент для bias
        
        l2_lambda = 0.001
        dw += l2_lambda * self.weights
        
        max_grad_norm = 1.0
        grad_norm = np.sqrt(np.sum(dw**2) + np.sum(db**2))
        if grad_norm > max_grad_norm:
            dw = dw * max_grad_norm / grad_norm
            db = db * max_grad_norm / grad_norm
        
        dx = np.dot(self.weights.T, dz)
        
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return dx