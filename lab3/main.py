# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 28.11.2025

import numpy as np
from neural_network import JordanElmanNetwork
from utils import generate_sequences, zscore_denormalize, zscore_normalize

SEQUENCE = "arithmetic"
SEQ_SIZE = 11
WIN_SIZE = 4
CTX_SIZE = 3
EFFECTOR_SIZE = 4
HIDDEN_SIZE = 3
LR = 1e-4

ALPHA_ELU = 1.0
EPOCHS = 1000000000
SAVE_CTX_EACH_EPOCH = True

MAX_ERROR = 1e-5
ZSCORE = True
if __name__ == "__main__":
    sequences = generate_sequences()
    
    raw_sequence = sequences[SEQUENCE] 

    input_raw = raw_sequence[:SEQ_SIZE]

    print(f'Обучающая выборка: {input_raw}\nВалидационная выборка: {raw_sequence[SEQ_SIZE:]}')

    if ZSCORE:
        processed_input, input_mean, input_std = zscore_normalize(input_raw)
    else:
        processed_input = np.array(input_raw)

    neural_network = JordanElmanNetwork(
        seq=processed_input,
        input_size=WIN_SIZE,
        hidden_size=HIDDEN_SIZE,
        context_size=CTX_SIZE,
        effector_size=EFFECTOR_SIZE,
        alpha=LR,
        max_errors=MAX_ERROR,
        max_iters=EPOCHS,
        predict_len=EFFECTOR_SIZE,
        reset_context=SAVE_CTX_EACH_EPOCH,
        effector_activation_type="elu",
        hidden_alpha=ALPHA_ELU
    )

    neural_network.train()

    predicted_processed = neural_network.predict()
    predicted = []
    if ZSCORE:
        predicted = zscore_denormalize(predicted_processed, input_mean, input_std)
    else:
        predicted = predicted_processed

    predict_len_actual = len(predicted)
    val_start = SEQ_SIZE
    validation_target = raw_sequence[val_start : val_start + predict_len_actual]

    for i in range(min(len(predicted), len(validation_target))):
        expected = validation_target[i]
        got = predicted[i]
        diff = got - expected
        status = "(equal)" if abs(diff) < 1e-6 else ""
        print(f"Ожидаемый результат: {expected}. Получено: {got};")

    