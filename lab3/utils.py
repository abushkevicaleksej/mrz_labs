###############################
# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциальной линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией вспомогательных функций
# Дата 28.11.2025

from typing import List

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

def normalize_sequence(sequence: List[float]) -> List[float]:
    min_val = min(sequence)
    max_val = max(sequence)
    
    if max_val == min_val:
        return [0.5] * len(sequence)
    
    return [(x - min_val) / (max_val - min_val) for x in sequence]

def denormalize_sequence(normalized: List[float], original: List[float]) -> List[float]:
    min_val = min(original)
    max_val = max(original)
    
    return [x * (max_val - min_val) + min_val for x in normalized]