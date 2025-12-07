from typing import List
import numpy as np
from fclayer import FCLayer
from activation_func import ELU, Linear

MAX_ERROR = 1e-6

class JordanElmanNetwork:
    
    def __init__(self, window_size: int, hidden_size: int, output_size: int, 
                 context_reset: bool = False, elu_alpha: float = 1.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_reset = context_reset
        
        self.context_elman = np.zeros((hidden_size, 1)) # P(t-1)
        self.context_jordan = np.zeros((output_size, 1)) # Y(t-1)
        
        input_total_size = window_size + hidden_size + output_size
        
        self.hidden_layer = FCLayer(input_total_size, hidden_size, ELU(alpha=elu_alpha))
        
        self.output_layer = FCLayer(hidden_size, output_size, Linear())
        
        self.loss_history = []
    
    def reset_context(self):
        self.context_elman = np.zeros((self.hidden_size, 1))
        self.context_jordan = np.zeros((self.output_size, 1))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        S(t) = W_x * X(t) + W_p * P(t-1) + W_y * Y(t-1)
        """
        combined_input = np.vstack([
            x.reshape(-1, 1),      # X(t)
            self.context_elman,    # P(t-1) - выход скрытого слоя на пред. шаге
            self.context_jordan    # Y(t-1) - выход сети на пред. шаге
        ])
        
        # P(t) = F(S(t))
        hidden_output = self.hidden_layer.forward(combined_input)
        
        # Y(t) = F_out(S_out(t))
        network_output = self.output_layer.forward(hidden_output)
        
        self.context_elman = hidden_output.copy()
        self.context_jordan = network_output.copy()
        
        return network_output.flatten()
    
    def backward(self, target: np.ndarray, learning_rate: float) -> float:
        """
        E = 1/2 * sum((y - target)^2)
        dE/dy = (y - target)
        """
        target = target.reshape(-1, 1)
        
        output_error = self.output_layer.output - target
        
        # dE/dy = (y - target)
        output_grad = output_error 
        
        hidden_grad = self.output_layer.backward(output_grad, learning_rate)
        
        _ = self.hidden_layer.backward(hidden_grad, learning_rate)
        
        sse = 0.5 * np.sum(output_error ** 2)
        return sse
    
    def train(self, sequence: List[float], epochs: int, learning_rate: float = 0.01) -> None:
        n = len(sequence)
        
        for epoch in range(epochs):
            if self.context_reset:
                self.reset_context()
            
            epoch_loss = 0
            
            for i in range(self.window_size, n - self.output_size + 1):
                input_window = np.array(sequence[i - self.window_size:i])
                target = np.array(sequence[i:i + self.output_size])
                
                self.forward(input_window)
                
                loss = self.backward(target, learning_rate)
                
                epoch_loss += loss
            
            if epoch % 1000 == 0:
                print(f"Эпоха {epoch:5d}, Суммарная ошибка: {epoch_loss:.8f}")

            if epoch_loss <= MAX_ERROR:
                print(f"\nОбучение остановлено на эпохе {epoch}, достигнута требуемая точность (ошибка <= {MAX_ERROR})")
                break
    
    def predict(self, initial_window: List[float], steps: int) -> List[float]:
        if len(initial_window) != self.window_size:
            raise ValueError(f"Начальное окно должно иметь размер {self.window_size}")
        
        predictions = []
        current_window = list(initial_window)
        
        for step in range(steps):
            window_array = np.array(current_window[-self.window_size:])
            
            prediction = self.forward(window_array)
            
            predictions.extend(prediction)
            
            if self.output_size == 1:
                current_window.append(prediction[0])
            else:
                current_window.append(prediction[-1])
        
        return predictions
    
    def evaluate(self, sequence: List[float]) -> float:
        self.reset_context()
        n = len(sequence)
        total_loss = 0
        
        for i in range(self.window_size, n - self.output_size + 1):
            input_window = np.array(sequence[i - self.window_size:i])
            target = np.array(sequence[i:i + self.output_size])
            
            prediction = self.forward(input_window)
            
            error = prediction - target.flatten()
            loss = 0.5 * np.sum(error ** 2)
            
            total_loss += loss
        
        return total_loss