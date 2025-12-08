from typing import Callable, Tuple
import numpy as np

class FCLayer:
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        self.weights = np.random.uniform(low=-1, high=1, size=(output_size, input_size))
        
        self.threshold = np.random.uniform(low=-0.1, high=0.1, size=(output_size, 1))
        
        self.activation = activation
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        S(t) = W * x(t) - T
        """
        input_col = x.reshape(-1, 1)
        
        z = np.dot(self.weights, input_col) - self.threshold
        
        output = self.activation.forward(z)
        return output, z
    
    def backward(self, output_gradient: np.ndarray, x_input: np.ndarray, z_pre_activation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        input_col = x_input.reshape(-1, 1)
        
        # dE/dS = dE/dy * F'(S) = delta
        deriv = self.activation._compute_derivative(z_pre_activation)
        delta = output_gradient * deriv
        
        # dE/dW = delta * x^T
        dw = np.dot(delta, input_col.T)
        
        # dE/dx = W^T * delta
        dx = np.dot(self.weights.T, delta)
        
        # S = Wx - T
        # dS/dT = -1
        # dE/dT = dE/dS * dS/dT = delta * (-1) = -delta
        dt = -delta
        
        return dx, dw, dt

    def update_weights(self, grad_w: np.ndarray, grad_t: np.ndarray, learning_rate: float):
        """
        W(t+1) = W(t) - lr * dE/dW
        T(t+1) = T(t) - lr * dE/dT
        """
        self.weights -= learning_rate * grad_w
        self.threshold -= learning_rate * grad_t