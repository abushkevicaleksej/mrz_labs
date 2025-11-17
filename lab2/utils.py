###############################
# Лабораторная работа №4 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Хопфилда с дискретным состоянием и дискретным временем в асинхронном режиме.
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл, содержащий реализацию вспомогательных функций
# Дата 17.11.2025

import numpy as np
import os
from neural_network import HopfieldNetwork
from image_processor import ImageProcessor

def create_training_set(file_paths, image_size=None):
    processor = ImageProcessor(image_size)
    patterns = []
    successful_files = []
    
    for file_path in file_paths:
        img = processor.load_image(file_path)
        if img is not None:
            vector = processor.image_to_vector(img)
            patterns.append(vector)
            successful_files.append(file_path)
        else:
            print(f"Не удалось загрузить изображение: {file_path}")
    
    if not patterns:
        raise ValueError("Не удалось загрузить ни одного изображения для обучения")
    
    return np.array(patterns), processor, successful_files

def test_symbol_recognition(training_files, test_file, noise_level=0.3, noise_type="both", image_size=None, max_iterations=1000):
    patterns, processor, successful_training_files = create_training_set(training_files, image_size)
    
    if image_size is None and processor.image_size:
        image_size = processor.image_size
    
    n_neurons = patterns.shape[1]
    
    network = HopfieldNetwork(n_neurons)
    
    print(f"Количество нейронов: {n_neurons}")
    print(f"Размер изображения: {processor.image_size}")
    print(f"Ёмкость сети составляет: {n_neurons / (4 * np.log(n_neurons)) if n_neurons > 1 else 0:.2f}")

    network.train(patterns)
    
    test_img = processor.load_image(test_file)
    
    test_name = os.path.basename(test_file)
    test_vector = processor.image_to_vector(test_img)
    
    noisy_vector = processor.add_noise(test_vector, noise_level, noise_type)
    
    print(f"\nРаспознавание изображения '{test_name}'")
    print(f"Максимальное количество итераций: {max_iterations}")
    
    result, iterations = network.predict(noisy_vector, max_iter=max_iterations)
    
    success = False
    matched_file = None
    for i, pattern in enumerate(patterns):
        if np.array_equal(result, pattern):
            matched_file = successful_training_files[i]
            matched_name = os.path.basename(matched_file)
            print(f"  Результат: распознан как '{matched_name}'")
            success = True
            break
    
    if not success:
        print("  Результат: не распознан (ложный аттрактор)")
    
    print(f"  Количество итераций до релаксации: {iterations}")
    
    processor.visualize_patterns(test_vector, noisy_vector, result, test_name, iterations)
    
    return result, iterations, success, matched_file

def get_absolute_path(relative_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)