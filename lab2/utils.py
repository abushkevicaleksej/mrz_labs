import numpy as np

from neural_network import HopfieldNetwork
from image_processor import ImageProcessor
import os

def create_training_set(symbols, image_size=(10, 10)):

    processor = ImageProcessor(image_size)
    patterns = []
    
    for symbol in symbols:
        if isinstance(symbol, str) and os.path.isfile(symbol):
            img = processor.load_image(symbol)
        else:
            img = processor.create_symbol_image(symbol)
        vector = processor.image_to_vector(img)
        patterns.append(vector)
    
    return np.array(patterns), processor


def test_symbol_recognition(symbols, test_symbol, noise_level=0.3, noise_type="both", image_size=(10, 10)):

    patterns, processor = create_training_set(symbols, image_size)
    n_neurons = patterns.shape[1]
    
    network = HopfieldNetwork(n_neurons)
    
    print(f"Количество нейронов: {n_neurons}")
    print(f"Ёмкость сети составляет: {n_neurons / 4 * np.log(n_neurons)}")

    network.train(patterns)
    
    print("Проверка устойчивости эталонов:")
    for i, symbol in enumerate(symbols):
        is_stable = network.check_stability(patterns[i])
        print(f"  '{symbol}': {'устойчив' if is_stable else 'неустойчив'}")

    if isinstance(test_symbol, str) and os.path.isfile(test_symbol):
        test_img = processor.load_image(test_symbol)
        test_name = os.path.basename(test_symbol)
    else:
        test_img = processor.create_symbol_image(test_symbol)
        test_name = test_symbol

    test_vector = processor.image_to_vector(test_img)
    
    noisy_vector = processor.add_noise(test_vector, noise_level, noise_type)
    
    print(f"\nРаспознавание символа '{test_name}' с шумом {noise_level*100}%:")
    result, iterations = network.predict(noisy_vector)
    
    success = False
    for i, pattern in enumerate(patterns):
        if np.array_equal(result, pattern):
            trained_name = os.path.basename(symbols[i]) if isinstance(symbols[i], str) and os.path.isfile(
                symbols[i]) else symbols[i]
            print(f"  Результат: распознан как '{trained_name}'")
            success = True
            break
    
    if not success:
        print("  Результат: не распознан (ложный аттрактор)")
    
    print(f"  Количество итераций: {iterations}")
    
    processor.visualize_patterns(test_vector, noisy_vector, result, test_name, iterations)
    
    return result, iterations
