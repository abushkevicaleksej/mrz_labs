# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 28.11.2025

import numpy as np
from typing import List, Tuple
from neural_network import JordanElmanNetwork
from utils import generate_sequences

WIN_SIZE = 3
LEARNING_RATE = 0.01 
STEPS = 3
HIDDEN_SIZE = 4
OUT_SIZE = 1
RESET_CTX = False
ALPHA = 1.0
EPOCHS = 100000

def train_test_split(sequence: List[float], test_size: int = 5) -> Tuple[List[float], List[float]]:
    if len(sequence) <= test_size:
        raise ValueError(f"Последовательность слишком короткая: {len(sequence)}, нужно минимум {test_size+WIN_SIZE}")
    
    train_seq = sequence[:-test_size]
    test_seq = sequence[-test_size:]
    
    print(f"Обучающая выборка: {len(train_seq)} элементов")
    print(f"Тестовая выборка: {len(test_seq)} элементов")
    
    return train_seq, test_seq

def scale_sequence(sequence: List[float]) -> Tuple[List[float], float, float]:
    epsilon = 1e-10
    min_val = min(sequence) - epsilon
    max_val = max(sequence) + epsilon
    
    if abs(max_val - min_val) < epsilon:
        return [0.5] * len(sequence), min_val, max_val
    
    scaled = [(x - min_val) / (max_val - min_val) for x in sequence]
    return scaled, min_val, max_val

def inverse_scale(scaled_values: List[float], min_val: float, max_val: float) -> List[float]:
    return [x * (max_val - min_val) + min_val for x in scaled_values]

def preprocess_sequence(sequence: List[float], method: str = 'scale') -> Tuple[List[float], dict]:
    if method == 'log':
        processed = [np.log1p(x) for x in sequence]
        params = {'method': 'log'}
        return processed, params
    elif method == 'scale':
        processed, min_val, max_val = scale_sequence(sequence)
        params = {'method': 'scale', 'min_val': min_val, 'max_val': max_val}
        return processed, params
    else:
        params = {'method': 'none'}
        return sequence.copy(), params

def inverse_preprocess(values: List[float], params: dict) -> List[float]:
    method = params.get('method', 'none')
    
    if method == 'log':
        return [np.expm1(x) for x in values]
    elif method == 'scale':
        min_val = params['min_val']
        max_val = params['max_val']
        return inverse_scale(values, min_val, max_val)
    else:
        return values

def main():
    sequences = generate_sequences()
    
    seq_name = "fibonacci"
    raw_sequence = sequences[seq_name]
    
    print("=" * 60)
    print(f"Тестирование на последовательности: {seq_name}")
    print(f"Исходная последовательность: {raw_sequence}")
    print(f"Длина последовательности: {len(raw_sequence)}")
    print("=" * 60)
    
    try:
        train_seq_raw, test_seq_raw = train_test_split(raw_sequence, test_size=STEPS)
        print(f"\nОбучающая выборка: {train_seq_raw}")
        print(f"Тестовая выборка: {test_seq_raw}")
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    
    # Предобработка
    train_seq, train_params = preprocess_sequence(train_seq_raw, method='scale')
    
    print(f"  Размер окна: {WIN_SIZE}")
    print(f"  Скрытый слой: {HIDDEN_SIZE} нейронов")
    print(f"  Выходной слой: {OUT_SIZE} нейронов")
    print(f"  Сброс контекста: {RESET_CTX}")
    print(f"  α (ELU): {ALPHA}")
    
    network = JordanElmanNetwork(
        window_size=WIN_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUT_SIZE,
        context_reset=RESET_CTX,
        elu_alpha=ALPHA
    )
    
    print(f"\nНачало обучения ({EPOCHS} эпох)...")
    network.train(train_seq, EPOCHS, LEARNING_RATE)
    
    train_loss = network.evaluate(train_seq)
    print(f"\nСуммарная ошибка на обучающей выборке: {train_loss:.6f}")
    
    initial_window = train_seq[-WIN_SIZE:]
    print(f"\nНачальное окно для прогноза (нормализованное): {initial_window}")
    
    predictions_scaled = network.predict(initial_window, STEPS)
    predictions = inverse_preprocess(predictions_scaled, train_params)
    
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
    print(f"{'='*60}")
    
    total_abs_error = 0
    total_relative_error = 0
    
    for i, (pred, actual) in enumerate(zip(predictions, test_seq_raw), 1):
        error = abs(pred - actual)
        relative_error = abs(pred - actual) / abs(actual) * 100 if actual != 0 else 0
        
        total_abs_error += error
        total_relative_error += relative_error
        
        print(f"Шаг {i}:")
        print(f"  Прогноз: {pred:.4f}")
        print(f"  Факт:    {actual:.4f}")
        print(f"  Абс. ошибка: {error:.4f}")
        print(f"  Отн. ошибка: {relative_error:.2f}%")
        print()
    
    if len(predictions) > 0:
        print(f"Средняя абсолютная ошибка: {total_abs_error / len(predictions):.4f}")
        print(f"Средняя относительная ошибка: {total_relative_error / len(predictions):.2f}%")
    
    return network

if __name__ == "__main__":
    trained_network = main()