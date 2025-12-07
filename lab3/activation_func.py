import numpy as np

class ELU:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.input = None
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """
        f(x) = x, если x >= 0
        f(x) = alpha * (exp(x) - 1), если x < 0
        """
        out = x.copy()
        neg_mask = x < 0
        
        out[neg_mask] = self.alpha * (np.exp(x[neg_mask]) - 1)
        
        return out

    def _compute_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        f'(x) = 1, если x >= 0
        f'(x) = alpha * exp(x), если x < 0
        """
        out = np.ones_like(x)
        neg_mask = x < 0
        
        out[neg_mask] = self.alpha * np.exp(x[neg_mask])
        
        return out
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x.copy()
        return self._compute_forward(x)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self._compute_derivative(self.input)


class Tanh:
    def __init__(self):
        self.output = None
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
        
    def _compute_derivative(self, output: np.ndarray) -> np.ndarray:
        # Производная tanh выражается через значение функции: 1 - y^2
        return 1 - output ** 2
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = self._compute_forward(x)
        return self.output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self._compute_derivative(self.output)


class Linear:
    def __init__(self):
        pass
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return x
        
    def _compute_derivative(self, x: np.ndarray) -> float:
        return 1.0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._compute_forward(x)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self._compute_derivative(output_gradient)