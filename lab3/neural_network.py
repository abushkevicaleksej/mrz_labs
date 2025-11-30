###############################
# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциальной линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией сети Джордана-Элмана
# Дата 28.11.2025

from typing import List

import numpy as np

from fclayer import FCLayer

from activation_func import Tanh, ELU, Linear

class JordanElmanNetwork:
    """Сеть Джордана-Элмана для прогнозирования последовательностей"""
    
    def __init__(self, window_size: int, hidden_size: int, output_size: int, 
                 context_reset: bool = False, elu_alpha: float = 1.0):
        # Параметры сети
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_reset = context_reset
        self.elu_alpha = elu_alpha
        
        # Контекстные нейроны
        self.context_elman = np.zeros((hidden_size, 1))  # Контекст Элмана
        self.context_jordan = np.zeros((output_size, 1))  # Контекст Джордана
        
        # Создание слоев
        # Входной слой: window_size + hidden_size (контекст Элмана) + output_size (контекст Джордана)
        input_total_size = window_size + hidden_size + output_size
        
        # Скрытый слой
        self.hidden_layer = FCLayer(input_total_size, hidden_size, Tanh())
        
        # Выходной слой (эффекторный) с ELU активацией
        self.output_layer = FCLayer(hidden_size, output_size, ELU(alpha=elu_alpha))
        
        # История для отладки
        self.loss_history = []
    
    def reset_context(self):
        """Сброс контекстных нейронов"""
        self.context_elman = np.zeros((self.hidden_size, 1))
        self.context_jordan = np.zeros((self.output_size, 1))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через сеть"""
        # Объединяем вход с контекстными нейронами
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
        """Обратное распространение ошибки и обновление весов"""
        # Преобразуем цель в столбец
        target = target.reshape(-1, 1)
        
        # Вычисляем градиент на выходном слое
        output_error = self.output_layer.output - target
        output_grad = 2 * output_error  # Производная MSE
        
        # Распространение ошибки через выходной слой
        hidden_grad = self.output_layer.backward(output_grad, learning_rate)
        
        # Распространение ошибки через скрытый слой
        # Для контекстных связей мы не обновляем веса, так как они фиксированы (единичные)
        _ = self.hidden_layer.backward(hidden_grad, learning_rate)
        
        # Вычисляем MSE
        mse = np.mean(output_error ** 2) / 2
        return mse
    
    def train(self, sequence: List[float], epochs: int, learning_rate: float = 0.01) -> None:
        """Обучение сети на последовательности"""
        n = len(sequence)
        
        for epoch in range(epochs):
            # Сброс контекста в начале эпохи, если указано
            if self.context_reset:
                self.reset_context()
            else:
                # Иначе сохраняем контекст с предыдущей эпохи
                pass
            
            total_loss = 0
            num_predictions = 0
            
            # Обучение на скользящем окне
            for i in range(self.window_size, n - self.output_size):
                # Входное окно
                input_window = sequence[i - self.window_size:i]
                
                # Целевые значения (может быть несколько, если output_size > 1)
                targets = sequence[i:i + self.output_size]
                
                # Прямой проход
                prediction = self.forward(np.array(input_window))
                
                # Обратное распространение
                loss = self.backward(np.array(targets), learning_rate)
                
                total_loss += loss
                num_predictions += 1
            
            # Средняя ошибка за эпоху
            avg_loss = total_loss / num_predictions if num_predictions > 0 else 0
            self.loss_history.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, MSE: {avg_loss:.6f}")
    
    def predict(self, initial_window: List[float], steps: int) -> List[float]:
        """Прогнозирование нескольких значений методом скользящего окна"""
        if len(initial_window) != self.window_size:
            raise ValueError(f"Начальное окно должно иметь размер {self.window_size}")
        
        # Сбрасываем контекст для чистого прогнозирования
        self.reset_context()
        
        predictions = []
        current_window = initial_window.copy()
        
        for step in range(steps):
            # Преобразуем в numpy array
            window_array = np.array(current_window)
            
            # Прямой проход
            prediction = self.forward(window_array)
            
            # Сохраняем прогноз
            predictions.extend(prediction)
            
            # Обновляем окно для следующего шага
            if self.output_size == 1:
                current_window = current_window[1:] + [prediction[0]]
            else:
                # Если прогнозируем несколько значений, используем последнее
                current_window = current_window[1:] + [prediction[-1]]
        
        return predictions
    
    def evaluate(self, sequence: List[float]) -> float:
        """Оценка качества модели на тестовой последовательности"""
        n = len(sequence)
        total_loss = 0
        num_predictions = 0
        
        # Сброс контекста для чистого тестирования
        self.reset_context()
        
        for i in range(self.window_size, n - self.output_size):
            # Входное окно
            input_window = sequence[i - self.window_size:i]
            
            # Целевые значения
            targets = sequence[i:i + self.output_size]
            
            # Прямой проход
            prediction = self.forward(np.array(input_window))
            
            # Вычисляем MSE
            error = prediction - np.array(targets)
            mse = np.mean(error ** 2)
            
            total_loss += mse
            num_predictions += 1
        
        return total_loss / num_predictions if num_predictions > 0 else 0