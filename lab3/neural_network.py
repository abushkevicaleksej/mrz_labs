from typing import List
import numpy as np
from fclayer import FCLayer
from activation_func import Tanh, ELU, Linear

class JordanElmanNetwork:
    
    def __init__(self, window_size: int, hidden_size: int, output_size: int, 
                 context_reset: bool = False, elu_alpha: float = 1.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_reset = context_reset
        
        self.context_elman = np.zeros((hidden_size, 1))
        self.context_jordan = np.zeros((output_size, 1))
        
        input_total_size = window_size + hidden_size + output_size
        
        # 1. Создаем слой с ELU
        self.hidden_layer = FCLayer(input_total_size, hidden_size, ELU(alpha=elu_alpha))
        
        # === КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: "Глушим" веса контекста ===
        # Структура входа: [Window Inputs | Context Elman | Context Jordan]
        # Мы оставляем веса для Window Inputs (первые window_size колонок) как есть (He init).
        # А веса для Context (остальные колонки) делаем очень маленькими.
        # Это позволяет сети сначала выучить прямую зависимость, не отвлекаясь на шум контекста.
        self.hidden_layer.weights[:, window_size:] *= 0.01
        # ====================================================

        # Выходной слой линейный
        self.output_layer = FCLayer(hidden_size, output_size, Linear())
        
        self.loss_history = []
    
    def reset_context(self):
        self.context_elman = np.zeros((self.hidden_size, 1))
        self.context_jordan = np.zeros((self.output_size, 1))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        combined_input = np.vstack([
            x.reshape(-1, 1),
            self.context_elman,
            self.context_jordan
        ])
        
        hidden_output = self.hidden_layer.forward(combined_input)
        self.context_elman = hidden_output.copy()
        
        network_output = self.output_layer.forward(hidden_output)
        self.context_jordan = network_output.copy()
        
        return network_output.flatten()
    
    def backward(self, target: np.ndarray, learning_rate: float) -> float:
        target = target.reshape(-1, 1)
        output_error = self.output_layer.output - target
        
        # Обновляем выходной слой
        hidden_grad = self.output_layer.backward(output_error, learning_rate)
        
        # Обновляем скрытый слой
        # Важно: градиенты для части входа, отвечающей за контекст, вычисляются,
        # но в этой реализации (truncated BPTT) они не идут дальше назад во времени.
        _ = self.hidden_layer.backward(hidden_grad, learning_rate)
        
        return np.mean(output_error ** 2)
    
    def train(self, sequence: List[float], epochs: int, learning_rate: float = 0.01) -> None:
        n = len(sequence)
        
        # Адаптивный learning rate
        current_lr = learning_rate
        
        for epoch in range(epochs):
            if self.context_reset:
                self.reset_context()
            
            total_loss = 0
            
            # Обучение на всей последовательности
            for i in range(self.window_size, n - self.output_size):
                input_window = sequence[i - self.window_size:i]
                targets = sequence[i:i + self.output_size]
                
                self.forward(np.array(input_window))
                loss = self.backward(np.array(targets), current_lr)
                total_loss += loss
            
            avg_loss = total_loss / (n - self.window_size)
            self.loss_history.append(avg_loss)
            
            # Простой планировщик скорости обучения:
            # Каждые 500 эпох уменьшаем скорость, чтобы точнее попасть в минимум
            if epoch > 0 and epoch % 500 == 0:
                current_lr *= 0.7
                
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, MSE: {avg_loss:.8f}, LR: {current_lr:.6f}")

    def predict(self, initial_window: List[float], steps: int) -> List[float]:
        # Для прогноза контекст лучше сбросить и "прогреть" заново или просто сбросить
        self.reset_context()
        
        current_window = list(initial_window)
        predictions = []
        
        for _ in range(steps):
            window_array = np.array(current_window[-self.window_size:])
            pred = self.forward(window_array)
            val = pred[0]
            predictions.append(val)
            current_window.append(val)
            
        return predictions

    def evaluate(self, sequence: List[float]) -> float:
        self.reset_context()
        n = len(sequence)
        total_loss = 0
        count = 0
        
        for i in range(self.window_size, n - self.output_size):
            input_window = sequence[i - self.window_size:i]
            targets = sequence[i:i + self.output_size]
            
            prediction = self.forward(np.array(input_window))
            error = prediction - np.array(targets)
            total_loss += np.sum(error ** 2)
            count += 1
        
        return total_loss / count if count > 0 else 0