# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией сети Джордана-Элмана
# Дата 28.11.2025

from typing import List
import numpy as np
from fclayer import FCLayer
from activation_func import Tanh, ELU
from config import MAX_ERROR
class JordanElmanNetwork:
    
    def __init__(self, window_size: int, hidden_size: int, output_size: int, 
                 context_reset: bool = False, elu_alpha: float = 1.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_reset = context_reset
        self.elu_alpha = elu_alpha
        
        self.context_elman = np.zeros((hidden_size, 1))
        self.context_jordan = np.zeros((output_size, 1))
        
        input_total_size = window_size + hidden_size + output_size
        
        self.hidden_layer = FCLayer(input_total_size, hidden_size, Tanh())
        
        self.output_layer = FCLayer(hidden_size, output_size, ELU(alpha=elu_alpha))
        
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
        output_grad = 2 * output_error
        
        hidden_grad = self.output_layer.backward(output_grad, learning_rate)
        
        _ = self.hidden_layer.backward(hidden_grad, learning_rate)
        
        # mse = 0.5 * np.mean(output_error ** 2)
        mse = np.sum(output_error ** 2)
        return mse
    
    def train(self, sequence: List[float], epochs: int, learning_rate: float = 0.01) -> None:
        n = len(sequence)

        for epoch in range(epochs):
            if self.context_reset:
                self.reset_context()
            else:
                pass
            
            total_loss = 0
            for i in range(self.window_size, n - self.output_size):
                input_window = sequence[i - self.window_size:i]
                
                targets = sequence[i:i + self.output_size]
                
                prediction = self.forward(np.array(input_window))
                
                loss = self.backward(np.array(targets), learning_rate)
                
                total_loss += loss

            self.loss_history.append(total_loss)
            
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, MSE: {total_loss:.6f}")

            if total_loss <= MAX_ERROR:
                print(f"\nОбучение остановлено на эпохе {epoch}, достигнута требуемая точность (ошибка <= {MAX_ERROR})")
                break
    
    def predict(self, initial_window: List[float], steps: int) -> List[float]:
        if len(initial_window) != self.window_size:
            raise ValueError(f"Начальное окно должно иметь размер {self.window_size}")
        
        self.reset_context()
        
        predictions = []
        current_window = initial_window.copy()
        
        for step in range(steps):
            window_array = np.array(current_window)
            
            prediction = self.forward(window_array)
            
            predictions.extend(prediction)
            
            if self.output_size == 1:
                current_window = current_window[1:] + [prediction[0]]
            else:
                current_window = current_window[1:] + [prediction[-1]]
        
        return predictions
    
    def evaluate(self, sequence: List[float]) -> float:
        n = len(sequence)
        total_loss = 0

        self.reset_context()
        
        for i in range(self.window_size, n - self.output_size):
            input_window = sequence[i - self.window_size:i]
            
            targets = sequence[i:i + self.output_size]
            
            prediction = self.forward(np.array(input_window))
            
            error = prediction - np.array(targets)
            # mse = 0.5 * np.mean(error ** 2)
            mse = np.sum(error ** 2)
            total_loss += mse

        return total_loss