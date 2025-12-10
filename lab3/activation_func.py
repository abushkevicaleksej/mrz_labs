# activation_func.py
import numpy as np

class ELU:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def activate(self, x: np.ndarray) -> np.ndarray:
        # f(x) = x if x > 0 else alpha * (exp(x) - 1)
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))
