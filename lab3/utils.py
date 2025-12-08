# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией вспомогательных функций
# Дата 28.11.2025
import numpy as np

def generate_sequences() -> dict:
    sequences = {}
    
    sequences["geometric"] = [1 / (2 ** i) for i in range(15)]
    
    sequences["harmonic"] = [1 / i for i in range(1, 16)]
    
    sequences["sine"] = [1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0]

    sequences["increasing_zeros"] = [1]
    zeros_count = 1
    for i in range(14):
        sequences["increasing_zeros"].extend([0] * zeros_count)
        sequences["increasing_zeros"].append(1)
        zeros_count += 1
    sequences["increasing_zeros"] = sequences["increasing_zeros"][:15]
    
    sequences["arithmetic"] = [i for i in range(1, 16)]
    
    sequences["squares"] = [i ** 2 for i in range(1, 16)]
    
    fib = [1, 2]
    for i in range(13):
        fib.append(fib[-1] + fib[-2])
    sequences["fibonacci"] = fib[:15]
    
    sequences["geometric_2"] = [2 ** i for i in range(15)]
    
    sequences["bell"] = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975][:10]
    
    factorial = [1]
    for i in range(1, 15):
        factorial.append(factorial[-1] * i)
    sequences["factorial"] = factorial
    
    return sequences

def log_transform_sequence(sequence):
    """
    Применяет логарифмическое преобразование к последовательности.
    Используется для последовательностей с экспоненциальным ростом (Фибоначчи).
    """
    return [np.log1p(x) for x in sequence]


def inverse_log_transform(sequence_log):
    """
    Обратное логарифмическое преобразование.
    """
    return [np.expm1(x) for x in sequence_log]

def normalize_sequence(sequence):
    """Нормализует последовательность в диапазон [0, 1]"""
    min_val = min(sequence)
    max_val = max(sequence)
    if max_val == min_val:
        return [0.5] * len(sequence), min_val, max_val
    normalized = [(x - min_val) / (max_val - min_val) for x in sequence]
    return normalized, min_val, max_val

def denormalize_value(value, min_val, max_val):
    """Денормализует значение обратно в исходный диапазон"""
    return value * (max_val - min_val) + min_val