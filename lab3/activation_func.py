import numpy as np

def activate_elu(x: np.ndarray, alpha = 1.0) -> np.ndarray:
    # f(x) = x if x >= 0 else alpha * (exp(x) - 1)
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def activate_elu_derivative(x: np.ndarray, alpha = 1.0) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha * np.exp(x))
