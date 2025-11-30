###############################
# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциальной линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 28.11.2025

from typing import List, Tuple, Optional, Callable

import numpy as np

from neural_network import JordanElmanNetwork
from utils import generate_sequences, normalize_sequence, denormalize_sequence

WIN_SIZE = 1
HIDDEN_SIZE = 10
OUT_SIZE = 1
RESET_CTX = True
ALPHA = 0.1

if __name__ == "__main__":
    sequences = generate_sequences()
    
    sequence = sequences["arithmetic"]
    
    print(f"Последовательность: {sequence}")
    
    # normalized_seq = normalize_sequence(sequence)
    
    network = JordanElmanNetwork(
        window_size=WIN_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUT_SIZE,
        context_reset=RESET_CTX,
        elu_alpha=ALPHA
    )
    
    network.train(sequence, epochs=50000, learning_rate=0.001)
    
    test_loss = network.evaluate(sequence)
    print(f"\nСредняя ошибка на тестовой выборке: {test_loss:.6f}")
    
    initial_window = sequence[:WIN_SIZE]
    predictions_normalized = network.predict(initial_window, steps=7)
    
    # predictions = denormalize_sequence(predictions_normalized, sequence)
    
    print(f"\nПрогнозируемые значения:")
    for i, pred in enumerate(predictions_normalized, 1):
        actual_idx = WIN_SIZE + i - 1
        if actual_idx < len(sequence):
            actual = sequence[actual_idx]
            error = abs(pred - actual)
            print(f"Шаг {i}: Прогноз = {pred:.4f}, Факт = {actual:.4f}, Ошибка = {error:.4f}")
        else:
            print(f"Шаг {i}: Прогноз = {pred:.4f}")
