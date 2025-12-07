from typing import Callable
import numpy as np

class FCLayer:
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(output_size, input_size))
        
        self.activation = activation
        self.input = None
        self.output = None
        self.z = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        y = F(Wx)
        """
        self.input = x.reshape(-1, 1)
        
        # S = W * X
        self.z = np.dot(self.weights, self.input)
        
        # y = F(S)
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        output_gradient: dE/dy
        Возвращает: dE/dx
        """
        # dE/dS = dE/dy * F'(S)
        dz = output_gradient * self.activation.backward(self.z)
        
        # dE/dW = dE/dS * x^T
        dw = np.dot(dz, self.input.T)
        
        # dE/dx = W^T * dE/dS
        dx = np.dot(self.weights.T, dz)
        
        # W(t+1) = W(t) - alpha * dE/dW
        self.weights -= learning_rate * dw
        
        return dx