# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 28.11.2025

from neural_network import JordanElmanNetwork
from utils import generate_sequences
from config import WIN_SIZE, HIDDEN_SIZE, OUT_SIZE, RESET_CTX, ALPHA, EPOCHS, LEARNING_RATE, STEPS
from utils import normalize_sequence, denormalize_value, log_transform_sequence, inverse_log_transform

if __name__ == "__main__":
    sequences = generate_sequences()

    raw_sequence = sequences["squares"]
    sequence, min_val, max_val = normalize_sequence(raw_sequence)
    print(f"Исходная последовательность: {raw_sequence}")
    print(f"Диапазон значений: от {min(raw_sequence)} до {max(raw_sequence)}")
    sequence = log_transform_sequence(raw_sequence)

    print(f"\nЛогарифмированная последовательность: {sequence}")
    print(f"Диапазон логарифмированных значений: от {min(sequence):.4f} до {max(sequence):.4f}")

    network = JordanElmanNetwork(
        window_size=WIN_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUT_SIZE,
        context_reset=RESET_CTX,
        elu_alpha=ALPHA
    )

    network.train(sequence, EPOCHS, LEARNING_RATE)

    test_loss = network.evaluate(sequence)
    print(f"\nСуммарная ошибка на тестовой выборке: {test_loss:.6}")

    initial_window = sequence[:WIN_SIZE]
    predictions_log = network.predict(initial_window, STEPS)

    predictions = inverse_log_transform(predictions_log)

    print(f"\nПрогнозируемые значения:")
    for i, pred in enumerate(predictions, 1):
        actual_idx = WIN_SIZE + i - 1
        if actual_idx < len(raw_sequence):
            actual = raw_sequence[actual_idx]
            error = abs(pred - actual)
            relative_error = abs(pred - actual) / actual * 100 if actual != 0 else 0
            print(
                f"Шаг {i}: Прогноз = {pred:.4f}, Факт = {actual:.4f}")
        else:
            print(f"Шаг {i}: Прогноз = {pred:.4f}")
