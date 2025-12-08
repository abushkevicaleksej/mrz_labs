import numpy as np
from typing import List, Optional, Dict, Tuple

WIN_SIZE = 3

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
    fib = [0, 1]
    for i in range(18):
        fib.append(fib[-1] + fib[-2])
    sequences["fibonacci"] = fib[:20]
    sequences["geometric_2"] = [2 ** i for i in range(15)]
    sequences["bell"] = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975][:10]
    factorial = [1]
    for i in range(1, 15):
        factorial.append(factorial[-1] * i)
    sequences["factorial"] = factorial
    sequences["user"] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return sequences

def normalize_sequence(sequence):
    arr = np.array(sequence)
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        std = 1.0
        
    normalized = (arr - mean) / std
    return normalized.tolist(), mean, std

def denormalize_value(value, mean, std):
    return value * std + mean

def train_test_split(sequence: List[float], test_size: int = 5) -> Tuple[List[float], List[float]]:
    if len(sequence) <= test_size:
        raise ValueError(f"Последовательность слишком короткая: {len(sequence)}, нужно минимум {test_size+WIN_SIZE}")
    
    train_seq = sequence[:-test_size]
    test_seq = sequence[-test_size:]
    
    print(f"Обучающая выборка: {len(train_seq)} элементов")
    print(f"Тестовая выборка: {len(test_seq)} элементов")
    
    return train_seq, test_seq

def preprocess_sequence(sequence: List[float]) -> Tuple[List[float], dict]:
    processed, mean, std = normalize_sequence(sequence)
    params = {'mean': mean, 'std': std}
    return processed, params

def inverse_preprocess(values: List[float], params: dict) -> List[float]:
    mean = params['mean']
    std = params['std']
    return [denormalize_value(x, mean, std) for x in values]