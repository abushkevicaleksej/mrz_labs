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
    
    return sequences

def zscore_normalize(seq):
    arr = np.array(seq)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0: std = 1.0
    norm = (arr - mean) / std
    return norm, mean, std

def zscore_denormalize(norm, mean, std):
    return np.array(norm) * std + mean