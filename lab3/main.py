import numpy as np
from typing import List, Tuple
from neural_network import JordanElmanNetwork
from utils import (
    generate_sequences, 
    train_test_split, 
    preprocess_sequence, 
    inverse_preprocess
)

WIN_SIZE = 3
LEARNING_RATE = 0.001
STEPS = 4
HIDDEN_SIZE = 4
OUT_SIZE = 1
RESET_CTX = True
ALPHA = 1.0
EPOCHS = 100000

def main():
    sequences = generate_sequences()
    
    seq_name = "bell"
    raw_sequence = sequences[seq_name]
    
    print("=" * 60)
    print(f"Тестирование на последовательности: {seq_name}")
    print(f"Исходная последовательность: {raw_sequence}")
    print("=" * 60)
    
    try:
        train_seq_raw, test_seq_raw = train_test_split(raw_sequence, test_size=STEPS)
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    
    train_seq, train_params = preprocess_sequence(train_seq_raw)
    
    print(f"  Размер окна: {WIN_SIZE}")
    print(f"  Скрытый слой: {HIDDEN_SIZE} нейронов")
    print(f"  Выходной слой: {OUT_SIZE} нейронов")
    
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
    print(f"\nСуммарная ошибка на обучающей выборке (MSE): {train_loss:.6f}")
    
    initial_window = train_seq[-WIN_SIZE:]
    
    predictions_scaled = network.predict(initial_window, STEPS)
    predictions = inverse_preprocess(predictions_scaled, train_params)
    
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
    print(f"{'='*60}")
    
    total_abs_error = 0
    total_relative_error = 0
    
    for i, (pred, actual) in enumerate(zip(predictions, test_seq_raw), 1):
        error = abs(pred - actual)
        if actual != 0:
            relative_error = abs(pred - actual) / abs(actual) * 100 
        else:
            relative_error = 0.0 if error < 1e-9 else 100.0
            
        total_abs_error += error
        total_relative_error += relative_error
        
        print(f"Шаг {i}:")
        print(f"  Прогноз: {pred:.4f}")
        print(f"  Факт:    {actual:.4f}")

if __name__ == "__main__":
    main()