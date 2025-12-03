###############################
# Лабораторная работа №4 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Хопфилда с дискретным состоянием и дискретным временем в асинхронном режиме.
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл для построения графиков анализа работы сети Хопфилда
# Дата 17.11.2025

import matplotlib.pyplot as plt
import numpy as np
from neural_network import HopfieldNetwork
from image_processor import ImageProcessor
from utils import create_training_set
import os

class Graphics:
    def __init__(self):
        self.fig_size = (15, 5)
        
    def plot_iterations_vs_image_size(self, training_files, base_size=(50, 50), noise_level=0.3):
        """
        График зависимости количества итераций от размера образа
        """
        sizes = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120]
        iterations_list = []
        
        for size in sizes:
            print(f"Тестирование размера {size}x{size}")
            image_size = (size, size)
            
            try:
                patterns, processor, _ = create_training_set(training_files, image_size)
                n_neurons = patterns.shape[1]
                
                network = HopfieldNetwork(n_neurons)
                network.train(patterns)
                
                # Используем первый образ для тестирования
                test_vector = patterns[0]
                noisy_vector = processor.add_noise(test_vector, noise_level, "both")
                
                _, iterations = network.predict(noisy_vector, max_iter=1000)
                iterations_list.append(iterations)
                
            except Exception as e:
                print(f"Ошибка для размера {size}: {e}")
                iterations_list.append(None)
        
        # Фильтруем неудачные попытки
        valid_sizes = [sizes[i] for i in range(len(sizes)) if iterations_list[i] is not None]
        valid_iterations = [iterations_list[i] for i in range(len(sizes)) if iterations_list[i] is not None]
        
        plt.figure(figsize=self.fig_size)
        plt.subplot(1, 3, 1)
        plt.plot(valid_sizes, valid_iterations, marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.xlabel('Размер изображения (пиксели)')
        plt.ylabel('Количество итераций')
        plt.title('Зависимость итераций от размера образа')
        plt.grid(True, alpha=0.3)
        
        return valid_sizes, valid_iterations
    
    def plot_iterations_vs_noise_level(self, training_files, image_size=(50, 50), test_file_index=0):
        """
        График зависимости количества итераций от процента зашумления
        """
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        iterations_list = []
        success_rates = []
        
        patterns, processor, _ = create_training_set(training_files, image_size)
        n_neurons = patterns.shape[1]
        
        network = HopfieldNetwork(n_neurons)
        network.train(patterns)
        
        test_vector = patterns[test_file_index]
        
        for noise_level in noise_levels:
            print(f"Тестирование уровня шума {noise_level}")
            success_count = 0
            total_tests = 5
            noise_iterations = []
            
            for _ in range(total_tests):
                noisy_vector = processor.add_noise(test_vector, noise_level, "both")
                result, iterations = network.predict(noisy_vector, max_iter=1000)
                
                if np.array_equal(result, test_vector):
                    success_count += 1
                noise_iterations.append(iterations)
            
            avg_iterations = np.mean(noise_iterations)
            iterations_list.append(avg_iterations)
            success_rates.append(success_count / total_tests * 100)
        
        plt.subplot(1, 3, 2)
        plt.plot(noise_levels, iterations_list, 'ro-', linewidth=2, markersize=6, label='Средние итерации')
        plt.xlabel('Уровень шума')
        plt.ylabel('Количество итераций')
        plt.title('Зависимость итераций от уровня шума')
        plt.grid(True, alpha=0.3)
        
        # Дополнительный график успешности распознавания
        plt.twinx()
        plt.plot(noise_levels, success_rates, 'g--', linewidth=2, markersize=6, label='Процент успеха')
        plt.ylabel('Процент успешного распознавания (%)')
        plt.legend()
        
        return noise_levels, iterations_list, success_rates
    
    def plot_iterations_vs_patterns_count(self, all_training_files, image_size=(50, 50), noise_level=0.3):
        """
        График зависимости количества итераций от количества образов
        """
        pattern_counts = [2, 4, 6, 8, 10]
        iterations_list = []
        stability_rates = []
        
        for count in pattern_counts:
            print(f"Тестирование с {count} образами")
            current_files = all_training_files[:count]
            
            try:
                patterns, processor, _ = create_training_set(current_files, image_size)
                n_neurons = patterns.shape[1]
                
                network = HopfieldNetwork(n_neurons)
                network.train(patterns)
                
                # Проверяем устойчивость и итерации для каждого образа
                pattern_iterations = []
                stable_count = 0
                
                for i, pattern in enumerate(patterns):
                    # Проверяем устойчивость
                    if network.check_stability(pattern):
                        stable_count += 1
                    
                    # Тестируем с шумом
                    noisy_vector = processor.add_noise(pattern, noise_level, "both")
                    _, iterations = network.predict(noisy_vector, max_iter=1000)
                    pattern_iterations.append(iterations)
                
                avg_iterations = np.mean(pattern_iterations)
                iterations_list.append(avg_iterations)
                stability_rates.append(stable_count / count * 100)
                
            except Exception as e:
                print(f"Ошибка для {count} образов: {e}")
                iterations_list.append(None)
                stability_rates.append(None)
        
        # Фильтруем неудачные попытки
        valid_counts = [pattern_counts[i] for i in range(len(pattern_counts)) if iterations_list[i] is not None]
        valid_iterations = [iterations_list[i] for i in range(len(pattern_counts)) if iterations_list[i] is not None]
        valid_stability = [stability_rates[i] for i in range(len(pattern_counts)) if stability_rates[i] is not None]
        
        plt.subplot(1, 3, 3)
        plt.plot(valid_counts, valid_iterations, 'mo-', linewidth=2, markersize=6, label='Средние итерации')
        plt.xlabel('Количество образов')
        plt.ylabel('Количество итераций')
        plt.title('Зависимость итераций от количества образов')
        plt.grid(True, alpha=0.3)
        
        # Дополнительный график устойчивости
        plt.twinx()
        plt.plot(valid_counts, valid_stability, 'c--', linewidth=2, markersize=6, label='Процент устойчивости')
        plt.ylabel('Процент устойчивых образов (%)')
        plt.legend()
        
        return valid_counts, valid_iterations, valid_stability
    
    def build_all_plots(self, training_files):
        """
        Построение всех трех графиков
        """
        plt.figure(figsize=self.fig_size)
        
        # График 1: Размер изображения
        sizes, iterations_size = self.plot_iterations_vs_image_size(training_files)
        
        # График 2: Уровень шума
        # noise_levels, iterations_noise, success_rates = self.plot_iterations_vs_noise_level(training_files)
        
        # График 3: Количество образов
        # pattern_counts, iterations_patterns, stability_rates = self.plot_iterations_vs_patterns_count(training_files)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'sizes': sizes,
            'iterations_size': iterations_size,
            # 'noise_levels': noise_levels,
            # 'iterations_noise': iterations_noise,
            # 'success_rates': success_rates,
            # 'pattern_counts': pattern_counts,
            # 'iterations_patterns': iterations_patterns,
            # 'stability_rates': stability_rates
        }

def get_absolute_path(relative_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

if __name__ == "__main__":
    # Пример использования
    training_files = [
        get_absolute_path('pics/1.bmp'),
        get_absolute_path('pics/2.bmp'),
        get_absolute_path('pics/3.bmp'),
        get_absolute_path('pics/4.bmp'),
        get_absolute_path('pics/5.bmp'),
        get_absolute_path('pics/6.bmp'),
        get_absolute_path('pics/7.bmp'),
        get_absolute_path('pics/8.bmp'),
        get_absolute_path('pics/9.bmp'),
        get_absolute_path('pics/0.bmp'),
    ]
    
    graphics = Graphics()
    results = graphics.build_all_plots(training_files)
    
    print("Результаты экспериментов:")
    print(f"Размеры изображений: {results['sizes']}")
    print(f"Итерации по размерам: {results['iterations_size']}")
    print(f"Уровни шума: {results['noise_levels']}")
    print(f"Итерации по шуму: {results['iterations_noise']}")
    print(f"Проценты успеха: {results['success_rates']}")
    print(f"Количество образов: {results['pattern_counts']}")
    print(f"Итерации по количеству образов: {results['iterations_patterns']}")
    print(f"Проценты устойчивости: {results['stability_rates']}")