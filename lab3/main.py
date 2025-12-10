# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 28.11.2025

import numpy as np
from neural_network import JordanElmanNetwork
from utils import generate_sequences, normalize_sequence, denormalize_value
from config import (
    WIN_SIZE, HIDDEN_SIZE, CONTEXT_SIZE, JORDAN_SIZE, OUT_SIZE, 
    RESET_CTX, ALPHA, EPOCHS, LEARNING_RATE, STEPS, MAX_ERROR, TRAIN_RATIO
)

if __name__ == "__main__":
    # 1. Генерация данных
    sequences = generate_sequences()
    
    # Выбираем последовательность (например, синус или геометрическую)
    # raw_sequence = sequences["geometric"] 
    raw_sequence = sequences["arithmetic"] 
    # raw_sequence = sequences["fibonacci"]

    print(f"Последовательность: {raw_sequence[:15]}...")
    
    # 2. Нормализация
    # Сети лучше работают с данными в диапазоне [0, 1] или [-1, 1]
    sequence, min_val, max_val = normalize_sequence(raw_sequence)
    print(f"Диапазон нормализации: min={min_val}, max={max_val}")

    # 3. Подготовка обучающей выборки (Windowing)
    X = []
    Y = []
    for i in range(len(sequence) - WIN_SIZE):
        X.append(sequence[i : i + WIN_SIZE])
        # Предсказываем следующий элемент за окном
        Y.append(sequence[i + WIN_SIZE]) 
    
    # 4. Разделение на Train и Test
    # Индекс, где заканчивается обучение и начинается тест
    # Используем TRAIN_RATIO из конфига или задаем вручную, чтобы оставить хвост для проверки
    split_idx = int(len(X) * TRAIN_RATIO)
    
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]
    
    print(f"\nВсего окон: {len(X)}")
    print(f"Обучающая выборка: {len(X_train)} окон")
    print(f"Тестовая выборка: {len(X_test)} окон")

    # 5. Инициализация сети
    network = JordanElmanNetwork(
        input_size=WIN_SIZE,
        hidden_size=HIDDEN_SIZE,
        context_size=CONTEXT_SIZE,
        jordan_size=JORDAN_SIZE,
        output_size=OUT_SIZE,
        elu_alpha=ALPHA
    )

    # 6. Обучение
    print("\n--- Начало обучения ---")
    network.train(X_train, Y_train, EPOCHS, LEARNING_RATE, MAX_ERROR, RESET_CTX)

    # 7. Прогноз и Сравнение
    print("\n--- Результаты прогнозирования на тестовой выборке ---")
    
    # Для честного прогноза берем последнее окно из ОБУЧАЮЩЕЙ выборки.
    # Сеть будет пытаться предсказать то, что находится в X_test/Y_test.
    # initial_window - это 'сырые' нормализованные данные перед началом теста
    
    # Находим индекс в исходной последовательности, соответствующий началу теста
    # split_idx соответствует началу X_test. 
    # Данные для первого предсказания лежат в sequence[split_idx : split_idx + WIN_SIZE] - это первый элемент X_test
    # Но predict работает рекурсивно. Чтобы проверить динамику, подадим окно ПЕРЕД тестом.
    
    start_predict_idx = split_idx
    initial_window = sequence[start_predict_idx : start_predict_idx + WIN_SIZE]
    
    # Сколько шагов предсказывать (не больше, чем есть тестовых данных)
    predict_steps = min(STEPS, len(raw_sequence) - (start_predict_idx + WIN_SIZE))
    
    # Получаем нормализованные предсказания
    predictions_scaled = network.predict(initial_window, predict_steps)

    # Вывод таблицы
    print(f"{'Шаг':<5} | {'Прогноз':<12} | {'Факт':<12} | {'Разница':<12}")
    print("-" * 50)

    for i in range(predict_steps):
        # Индекс фактического значения в исходном массиве
        # initial_window заканчивается на (start_predict_idx + WIN_SIZE - 1)
        # первый прогноз (i=0) должен соответствовать (start_predict_idx + WIN_SIZE)
        actual_idx = start_predict_idx + WIN_SIZE + i
        
        if actual_idx >= len(raw_sequence):
            break
            
        # Денормализация прогноза
        pred_val = denormalize_value(predictions_scaled[i], min_val, max_val)
        
        # Фактическое значение (берем из raw_sequence)
        actual_val = raw_sequence[actual_idx]
        
        # Разница
        diff = pred_val - actual_val
        
        print(f"{i+1:<5} | {pred_val:<12.6f} | {actual_val:<12.6f} | {diff:<12.6f}")
