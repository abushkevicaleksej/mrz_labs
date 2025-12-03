from typing import List
import numpy as np
from fclayer import FCLayer
from activation_func import Tanh, ELU, Linear
import random
random.seed(42)
np.random.seed(42)

class JordanElmanNetwork:
    
    def __init__(self, window_size: int, hidden_size: int, output_size: int, 
                 context_reset: bool = False, elu_alpha: float = 1.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_reset = context_reset
        
        # Контекстные нейроны
        self.context_elman = np.zeros((hidden_size, 1))
        self.context_jordan = np.zeros((output_size, 1))
        
        # Общий размер входа
        input_total_size = window_size + hidden_size + output_size
        
        # Скрытый слой с ELU активацией
        self.hidden_layer = FCLayer(input_total_size, hidden_size, ELU(alpha=elu_alpha))
        
        # Выходной слой с линейной активацией
        self.output_layer = FCLayer(hidden_size, output_size, Linear())
        
        # История обучения
        self.loss_history = []
        
        # Для отслеживания градиентов
        self.gradient_norms = []
    
    def reset_context(self):
        """Сброс контекстных нейронов"""
        self.context_elman = np.zeros((self.hidden_size, 1))
        self.context_jordan = np.zeros((self.output_size, 1))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через сеть"""
        # Объединяем входное окно с контекстными нейронами
        combined_input = np.vstack([
            x.reshape(-1, 1),  # Входное окно
            self.context_elman,  # Контекст Элмана
            self.context_jordan  # Контекст Джордана
        ])
        
        # Прямой проход через скрытый слой
        hidden_output = self.hidden_layer.forward(combined_input)
        
        # Обновляем контекст Элмана
        self.context_elman = hidden_output.copy()
        
        # Прямой проход через выходной слой
        network_output = self.output_layer.forward(hidden_output)
        
        # Обновляем контекст Джордана
        self.context_jordan = network_output.copy()
        
        return network_output.flatten()
    
    def backward(self, target: np.ndarray, learning_rate: float) -> float:
        """Обратное распространение ошибки"""
        target = target.reshape(-1, 1)
        
        # Вычисляем ошибку на выходе
        output_error = self.output_layer.output - target
        
        # Градиент для выходного слоя
        output_grad = 2 * output_error / target.shape[0]  # Нормализованный градиент
        
        # Обратное распространение через выходной слой
        hidden_grad = self.output_layer.backward(output_grad, learning_rate)
        
        # Обратное распространение через скрытый слой
        _ = self.hidden_layer.backward(hidden_grad, learning_rate)
        
        # Вычисляем MSE
        mse = np.mean(output_error ** 2)
        return mse
    
    def train(self, sequence: List[float], epochs: int, learning_rate: float = 0.01) -> None:
        """Обучение сети"""
        n = len(sequence)
        
        # Начальная скорость обучения
        current_lr = learning_rate
        
        for epoch in range(epochs):
            # Сброс контекста в начале эпохи, если требуется
            if self.context_reset:
                self.reset_context()
            
            epoch_loss = 0
            num_predictions = 0
            
            # Обучение на скользящем окне
            for i in range(self.window_size, n - self.output_size + 1):
                # Входное окно
                input_window = sequence[i - self.window_size:i]
                
                # Целевые значения
                targets = sequence[i:i + self.output_size]
                
                # Прямой проход
                self.forward(np.array(input_window))
                
                # Обратное распространение
                loss = self.backward(np.array(targets), current_lr)
                
                epoch_loss += loss
                num_predictions += 1
            
            # Средняя ошибка за эпоху
            avg_loss = epoch_loss / num_predictions if num_predictions > 0 else 0
            self.loss_history.append(avg_loss)
            
            # Адаптивная скорость обучения
            if epoch > 0 and epoch % 100 == 0:
                # Уменьшаем скорость обучения, если ошибка не уменьшается
                if len(self.loss_history) > 10:
                    recent_improvement = self.loss_history[-10] - avg_loss
                    if recent_improvement < 1e-6:
                        current_lr *= 0.99
            
            # Вывод прогресса
            if epoch % 500 == 0:
                print(f"Эпоха {epoch:5d}, MSE: {avg_loss:.8f}, LR: {current_lr:.6f}")
    
    def predict(self, initial_window: List[float], steps: int) -> List[float]:
        """Прогнозирование нескольких значений вперед"""
        if len(initial_window) != self.window_size:
            raise ValueError(f"Начальное окно должно иметь размер {self.window_size}")
        
        predictions = []
        current_window = list(initial_window)
        
        for step in range(steps):
            # Преобразуем в numpy array
            window_array = np.array(current_window[-self.window_size:])
            
            # Прямой проход
            prediction = self.forward(window_array)
            
            # Сохраняем прогноз
            predictions.extend(prediction)
            
            # Обновляем окно для следующего шага
            if self.output_size == 1:
                current_window.append(prediction[0])
            else:
                # Если прогнозируем несколько значений, используем последнее
                current_window.append(prediction[-1])
        
        return predictions
    
    def evaluate(self, sequence: List[float]) -> float:
        """Оценка качества модели"""
        self.reset_context()
        n = len(sequence)
        total_loss = 0
        num_predictions = 0
        
        for i in range(self.window_size, n - self.output_size + 1):
            # Входное окно
            input_window = sequence[i - self.window_size:i]
            
            # Целевые значения
            targets = sequence[i:i + self.output_size]
            
            # Прямой проход
            prediction = self.forward(np.array(input_window))
            
            # Вычисляем ошибку
            error = prediction - np.array(targets)
            mse = np.mean(error ** 2)
            
            total_loss += mse
            num_predictions += 1
        
        return total_loss / num_predictions if num_predictions > 0 else 0