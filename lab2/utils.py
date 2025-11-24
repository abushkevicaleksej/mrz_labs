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
    
    print(f'Всего тренировочных образов: {len(file_paths)}')
    return np.array(patterns), processor, successful_files

def check_orthogonality(patterns):
    n_patterns = patterns.shape[0]
    print("Матрица скалярных произведений:")
    for i in range(n_patterns):
        row = []
        for j in range(n_patterns):
            dot_product = np.dot(patterns[i], patterns[j])
            row.append(dot_product)
        # print(f"  {row}")
    
    orthogonal_pairs = 0
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            if abs(np.dot(patterns[i], patterns[j])) < 1e-10:
                orthogonal_pairs += 1
                # print(f"Эталоны {i} и {j} ортогональны")
    
    print(f"Ортогональных пар: {orthogonal_pairs}/{n_patterns*(n_patterns-1)//2}")

def create_hadamard_patterns(n):
    """
    Создает ортогональные эталонные образы используя матрицу Адамара.
    Возвращает до n ортогональных векторов из ±1.
    """
    # Находим размер матрицы Адамара (должен быть степенью 2)
    size = 1
    while size < n:
        size *= 2
    
    # Строим матрицу Адамара рекурсивно
    H = np.array([[1]])
    while H.shape[0] < size:
        H = np.block([[H, H], [H, -H]])
    
    # Берем первые n строк (эталонов)
    patterns = H[:n, :]
    
    return patterns

def test_symbol_recognition(training_files, test_file, noise_level=0.3, noise_type="both", image_size=None, max_iterations=1000):
    patterns, processor, successful_training_files = create_training_set(training_files, image_size)
    
    if image_size is None and processor.image_size:
        image_size = processor.image_size
    
    n_neurons = patterns.shape[1]
    
    network = HopfieldNetwork(n_neurons)
    
    capacity = n_neurons - 1

    print(f"Количество нейронов: {n_neurons}")
    print(f"Размер изображения: {processor.image_size}")
    print(f"Ёмкость сети составляет: {capacity if n_neurons > 1 else 0:.2f}")

    if capacity < len(patterns):
        print(f"Количество образов превышает ёмкость сети: {capacity:.2f} < {len(patterns)}")
        # exit(1)

    network.train(patterns)

    print("Проверка устойчивости эталонов:")
    for i, file_path in enumerate(successful_training_files):
        is_stable = network.check_stability(patterns[i])
        print(f"  '{os.path.basename(file_path)}': {'устойчив' if is_stable else 'неустойчив'}")
        print("\n")
    
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