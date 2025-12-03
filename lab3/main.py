# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 28.11.2025

import numpy as np
from typing import List, Tuple
from neural_network import JordanElmanNetwork
from utils import generate_sequences
import random
random.seed(42)
np.random.seed(42)

WIN_SIZE = 3
LEARNING_RATE = 0.0001
STEPS = 3
HIDDEN_SIZE = 4
OUT_SIZE = 1
RESET_CTX = False
ALPHA = 1.0
EPOCHS = 100000


def train_test_split(sequence: List[float], test_size: int = 5) -> Tuple[List[float], List[float]]:
    """
    Разделение последовательности на обучающую и тестовую части
    
    Args:
        sequence: Исходная последовательность
        test_size: Количество элементов для тестирования
        
    Returns:
        train_seq: Обучающая последовательность
        test_seq: Тестовая последовательность
    """
    if len(sequence) <= test_size:
        raise ValueError(f"Последовательность слишком короткая: {len(sequence)}, нужно минимум {test_size+WIN_SIZE}")
    
    train_seq = sequence[:-test_size]
    test_seq = sequence[-test_size:]
    
    print(f"Обучающая выборка: {len(train_seq)} элементов")
    print(f"Тестовая выборка: {len(test_seq)} элементов")
    
    return train_seq, test_seq

def scale_sequence(sequence: List[float]) -> Tuple[List[float], float, float]:
    """
    Масштабирование последовательности в диапазон [0, 1]
    Более стабильное, чем логарифмирование
    
    Args:
        sequence: Исходная последовательность
        
    Returns:
        scaled: Масштабированная последовательность
        min_val: Минимальное значение
        max_val: Максимальное значение
    """
    # Добавляем небольшое значение для стабильности
    epsilon = 1e-10
    min_val = min(sequence) - epsilon
    max_val = max(sequence) + epsilon
    
    if abs(max_val - min_val) < epsilon:
        return [0.5] * len(sequence), min_val, max_val
    
    scaled = [(x - min_val) / (max_val - min_val) for x in sequence]
    return scaled, min_val, max_val

def inverse_scale(scaled_values: List[float], min_val: float, max_val: float) -> List[float]:
    """
    Обратное масштабирование к исходному диапазону
    
    Args:
        scaled_values: Масштабированные значения
        min_val: Минимальное значение исходной последовательности
        max_val: Максимальное значение исходной последовательности
        
    Returns:
        Обратно масштабированные значения
    """
    return [x * (max_val - min_val) + min_val for x in scaled_values]

def preprocess_sequence(sequence: List[float], method: str = 'scale') -> Tuple[List[float], dict]:
    """
    Предобработка последовательности с выбором метода
    
    Args:
        sequence: Исходная последовательность
        method: Метод предобработки ('scale', 'log', 'none')
        
    Returns:
        processed: Обработанная последовательность
        params: Параметры преобразования для обратного преобразования
    """
    if method == 'log':
        # Для последовательностей с экспоненциальным ростом
        processed = [np.log1p(x) for x in sequence]
        params = {'method': 'log'}
        return processed, params
    elif method == 'scale':
        # Для последовательностей с линейным/полиномиальным ростом
        processed, min_val, max_val = scale_sequence(sequence)
        params = {'method': 'scale', 'min_val': min_val, 'max_val': max_val}
        return processed, params
    else:
        # Без преобразования
        params = {'method': 'none'}
        return sequence.copy(), params

def inverse_preprocess(values: List[float], params: dict) -> List[float]:
    """
    Обратное преобразование предсказаний
    
    Args:
        values: Обработанные значения
        params: Параметры преобразования
        
    Returns:
        Обратно преобразованные значения
    """
    method = params.get('method', 'none')
    
    if method == 'log':
        return [np.expm1(x) for x in values]
    elif method == 'scale':
        min_val = params['min_val']
        max_val = params['max_val']
        return inverse_scale(values, min_val, max_val)
    else:
        return values

def warm_up_context(network: JordanElmanNetwork, sequence: List[float]) -> None:
    """
    Прогрев контекстных нейронов на известной последовательности
    
    Args:
        network: Нейронная сеть
        sequence: Известная последовательность для прогрева
    """
    if len(sequence) < WIN_SIZE:
        return
    
    network.reset_context()
    
    # Пропускаем несколько шагов через сеть для прогрева контекста
    for i in range(len(sequence) - WIN_SIZE):
        input_window = sequence[i:i + WIN_SIZE]
        _ = network.forward(np.array(input_window))

def main():
    """Основная функция с исправленным пайплайном"""
    # Генерация последовательностей
    sequences = generate_sequences()
    
    # Выбор последовательности для тестирования
    seq_name = "fibonacci"
    raw_sequence = sequences[seq_name]
    
    print("=" * 60)
    print(f"Тестирование на последовательности: {seq_name}")
    print(f"Исходная последовательность: {raw_sequence}")
    print(f"Длина последовательности: {len(raw_sequence)}")
    print("=" * 60)
    
    # Разделение на обучающую и тестовую выборки
    try:
        train_seq_raw, test_seq_raw = train_test_split(raw_sequence, test_size=STEPS)
        print(f"\nОбучающая выборка: {train_seq_raw}")
        print(f"Тестовая выборка: {test_seq_raw}")
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    
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
    
    # Обучение сети
    print(f"\nНачало обучения ({EPOCHS} эпох)...")
    network.train(train_seq, EPOCHS, LEARNING_RATE)
    
    # Оценка на обучающей выборке
    train_loss = network.evaluate(train_seq)
    print(f"\nОшибка на обучающей выборке: {train_loss:.6f}")
    
    # Прогрев контекста на последних элементах обучающей выборки
    warm_up_context(network, train_seq[-WIN_SIZE*2:])
    
    # Прогнозирование на тестовой выборке
    # Берем последние WIN_SIZE элементов обучающей выборки как начальное окно
    initial_window = train_seq[-WIN_SIZE:]
    print(f"\nНачальное окно для прогноза: {initial_window}")
    
    # Делаем прогнозы
    predictions_scaled = network.predict(initial_window, STEPS)
    
    # Обратное преобразование прогнозов
    predictions = inverse_preprocess(predictions_scaled, train_params)
    
    # Вывод результатов
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
    print(f"{'='*60}")
    
    total_error = 0
    total_relative_error = 0
    
    for i, (pred, actual) in enumerate(zip(predictions, test_seq_raw), 1):
        error = abs(pred - actual)
        relative_error = abs(pred - actual) / actual * 100 if actual != 0 else 0
        
        total_error += error
        total_relative_error += relative_error
        
        print(f"Шаг {i}:")
        print(f"  Прогноз: {pred:.4f}")
        print(f"  Факт: {actual:.4f}")
        print(f"  Абсолютная ошибка: {error:.4f}")
        print(f"  Относительная ошибка: {relative_error:.2f}%")
        print()
    
    avg_abs_error = total_error / len(predictions)
    avg_rel_error = total_relative_error / len(predictions)
    
    print(f"Средняя абсолютная ошибка: {avg_abs_error:.4f}")
    print(f"Средняя относительная ошибка: {avg_rel_error:.2f}%")
    
    return network

if __name__ == "__main__":
    trained_network = main()