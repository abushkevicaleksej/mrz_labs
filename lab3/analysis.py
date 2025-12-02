import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from matplotlib import cm
import time

# Для корректного отображения графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class NetworkVisualizer:
    """Класс для визуализации различных аспектов работы сети Джордана-Элмана"""
    
    def __init__(self, network_class):
        """
        Инициализация визуализатора
        
        Args:
            network_class: Класс нейронной сети (JordanElmanNetwork)
        """
        self.network_class = network_class
        self.results = {}
    
    def plot_training_history(self, network, title="График зависимости количества итераций обучения от среднеквадратичной ошибки", save_path=None):
        """
        График истории обучения (ошибка vs итерации)
        
        Args:
            network: Обученная нейронная сеть
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        if not network.loss_history:
            print("Нет данных об истории обучения")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Основной график ошибки
        epochs = range(1, len(network.loss_history) + 1)
        plt.plot(epochs, network.loss_history, 'b-', linewidth=2, label='MSE')
        
        # Добавляем скользящее среднее для сглаживания
        window_size = max(1, len(network.loss_history) // 20)
        if window_size > 1:
            moving_avg = np.convolve(network.loss_history, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            plt.plot(epochs[window_size-1:], moving_avg, 'r--', 
                    linewidth=2, label=f'Скользящее среднее (окно={window_size})')
        
        plt.xlabel('Итерация обучения', fontsize=12)
        plt.ylabel('Среднеквадратичная ошибка', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        # plt.yscale('log')  # Логарифмическая шкала для лучшей видимости
        
        # Добавляем аннотации
        min_loss = min(network.loss_history)
        min_epoch = network.loss_history.index(min_loss) + 1
        plt.annotate(f'Минимум: {min_loss:.4f}\nна эпохе {min_epoch}',
                    xy=(min_epoch, min_loss),
                    xytext=(min_epoch + len(epochs)*0.1, min_loss * 2),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=10)
        
        # Добавляем информацию о параметрах сети
        params_text = f"Размер окна: {network.window_size}\n"
        params_text += f"Скрытый слой: {network.hidden_size}\n"
        params_text += f"Выходной слой: {network.output_size}\n"
        params_text += f"α (ELU): {network.elu_alpha}"
        
        plt.text(0.02, 0.98, params_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_window_size_impact(self, sequence: List[float], 
                              hidden_size: int = 5,
                              output_size: int = 1,
                              window_sizes: List[int] = None,
                              epochs: int = 500,
                              learning_rate: float = 0.01,
                              title="Влияние размера окна на качество обучения",
                              save_path=None):
        """
        График влияния размера скользящего окна на ошибку
        
        Args:
            sequence: Исходная последовательность
            hidden_size: Размер скрытого слоя
            output_size: Размер выходного слоя
            window_sizes: Список размеров окон для тестирования
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        if window_sizes is None:
            window_sizes = [1, 2, 3, 4, 5, 6, 7]
        
        # Нормализация последовательности
        seq_min = min(sequence)
        seq_max = max(sequence)
        normalized_seq = [(x - seq_min) / (seq_max - seq_min) for x in sequence]
        
        final_losses = []
        training_times = []
        
        for window_size in window_sizes:
            if window_size >= len(sequence) - output_size:
                print(f"Размер окна {window_size} слишком большой для последовательности длины {len(sequence)}")
                continue
            
            print(f"Тестирование размера окна: {window_size}")
            
            # Создание и обучение сети
            network = self.network_class(
                window_size=window_size,
                hidden_size=hidden_size,
                output_size=output_size,
                context_reset=False,
                elu_alpha=1.0
            )
            
            # Измеряем время обучения
            start_time = time.time()
            network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
            training_time = time.time() - start_time
            
            # Оценка качества
            test_loss = network.evaluate(normalized_seq)
            
            final_losses.append(test_loss)
            training_times.append(training_time)
            
            print(f"  Ошибка: {test_loss:.6f}, Время обучения: {training_time:.2f} сек")
        
        # Построение графика
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График 1: Ошибка vs Размер окна
        ax1.plot(window_sizes[:len(final_losses)], final_losses, 'bo-', 
                linewidth=2, markersize=8, label='Финальная MSE')
        ax1.set_xlabel('Размер скользящего окна', fontsize=12)
        ax1.set_ylabel('Среднеквадратичная ошибка (MSE)', fontsize=12)
        ax1.set_title(f'{title}\n(скрытый слой: {hidden_size}, эпох: {epochs})', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Добавляем аннотации для минимальной ошибки
        if final_losses:
            min_loss_idx = np.argmin(final_losses)
            min_loss = final_losses[min_loss_idx]
            best_window = window_sizes[min_loss_idx]
            ax1.annotate(f'Оптимально: окно={best_window}\nMSE={min_loss:.4f}',
                        xy=(best_window, min_loss),
                        xytext=(best_window + 1, min_loss * 1.5),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=10)
        
        # График 2: Время обучения vs Размер окна
        ax2.bar(window_sizes[:len(training_times)], training_times, 
               color='orange', alpha=0.7)
        ax2.set_xlabel('Размер скользящего окна', fontsize=12)
        ax2.set_ylabel('Время обучения (сек)', fontsize=12)
        ax2.set_title('Время обучения в зависимости от размера окна', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения поверх столбцов
        for i, (ws, tt) in enumerate(zip(window_sizes[:len(training_times)], training_times)):
            ax2.text(ws, tt + max(training_times)*0.01, f'{tt:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Сохраняем результаты
        self.results['window_size_impact'] = {
            'window_sizes': window_sizes[:len(final_losses)],
            'losses': final_losses,
            'training_times': training_times
        }
    
    def plot_hidden_size_impact(self, sequence: List[float],
                              window_size: int = 3,
                              output_size: int = 1,
                              hidden_sizes: List[int] = None,
                              epochs: int = 300,
                              learning_rate: float = 0.01,
                              title="Влияние размера скрытого слоя на качество обучения",
                              save_path=None):
        """
        График влияния размера скрытого слоя на ошибку
        
        Args:
            sequence: Исходная последовательность
            window_size: Размер скользящего окна
            output_size: Размер выходного слоя
            hidden_sizes: Список размеров скрытого слоя для тестирования
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        if hidden_sizes is None:
            hidden_sizes = [1, 2, 3, 5, 8, 10, 15, 20]
        
        # Нормализация последовательности
        seq_min = min(sequence)
        seq_max = max(sequence)
        normalized_seq = [(x - seq_min) / (seq_max - seq_min) for x in sequence]
        
        final_losses = []
        convergence_speeds = []  # Эпоха, на которой ошибка опустилась ниже порога
        training_times = []
        
        for hidden_size in hidden_sizes:
            print(f"Тестирование размера скрытого слоя: {hidden_size}")
            
            # Создание и обучение сети
            network = self.network_class(
                window_size=window_size,
                hidden_size=hidden_size,
                output_size=output_size,
                context_reset=False,
                elu_alpha=1.0
            )
            
            # Измеряем время обучения
            start_time = time.time()
            network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
            training_time = time.time() - start_time
            
            # Оценка качества
            test_loss = network.evaluate(normalized_seq)
            
            # Определяем скорость сходимости (эпоха, на которой ошибка стала меньше 0.1)
            convergence_epoch = None
            for i, loss in enumerate(network.loss_history):
                if loss < 0.1:
                    convergence_epoch = i + 1
                    break
            
            final_losses.append(test_loss)
            convergence_speeds.append(convergence_epoch if convergence_epoch else epochs)
            training_times.append(training_time)
            
            print(f"  Ошибка: {test_loss:.6f}, Время: {training_time:.2f} сек")
        
        # Построение графиков
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 15))
        
        # График 1: Ошибка vs Размер скрытого слоя
        ax1.plot(hidden_sizes[:len(final_losses)], final_losses, 'go-',
                linewidth=2, markersize=8, label='Финальная MSE')
        ax1.set_xlabel('Размер скрытого слоя', fontsize=12)
        ax1.set_ylabel('Среднеквадратичная ошибка (MSE)', fontsize=12)
        ax1.set_title(f'{title}',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Сохраняем результаты
        self.results['hidden_size_impact'] = {
            'hidden_sizes': hidden_sizes[:len(final_losses)],
            'losses': final_losses,
            'convergence_speeds': convergence_speeds,
            'training_times': training_times
        }
    
    def plot_3d_parameter_surface(self, sequence: List[float],
                                window_sizes: List[int],
                                hidden_sizes: List[int],
                                output_size: int = 1,
                                epochs: int = 200,
                                learning_rate: float = 0.01,
                                title="Поверхность ошибки в зависимости от параметров сети",
                                save_path=None):
        """
        3D поверхность ошибки в зависимости от размера окна и скрытого слоя
        
        Args:
            sequence: Исходная последовательность
            window_sizes: Список размеров окон для тестирования
            hidden_sizes: Список размеров скрытого слоя для тестирования
            output_size: Размер выходного слоя
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        # Нормализация последовательности
        seq_min = min(sequence)
        seq_max = max(sequence)
        normalized_seq = [(x - seq_min) / (seq_max - seq_min) for x in sequence]
        
        # Создаем сетку параметров
        X, Y = np.meshgrid(window_sizes, hidden_sizes)
        Z = np.zeros_like(X, dtype=float)
        training_times = np.zeros_like(X, dtype=float)
        
        print("Начало 3D анализа параметров...")
        total_tests = len(window_sizes) * len(hidden_sizes)
        current_test = 0
        
        for i, window_size in enumerate(window_sizes):
            for j, hidden_size in enumerate(hidden_sizes):
                current_test += 1
                print(f"Прогресс: {current_test}/{total_tests} "
                      f"({current_test/total_tests*100:.1f}%) - "
                      f"Окно: {window_size}, Скрытый: {hidden_size}")
                
                if window_size >= len(sequence) - output_size:
                    Z[j, i] = np.nan
                    training_times[j, i] = np.nan
                    continue
                
                # Создание и обучение сети
                network = self.network_class(
                    window_size=window_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    context_reset=False,
                    elu_alpha=1.0
                )
                
                # Обучение с измерением времени
                start_time = time.time()
                network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
                training_time = time.time() - start_time
                
                # Оценка качества
                test_loss = network.evaluate(normalized_seq)
                
                Z[j, i] = test_loss
                training_times[j, i] = training_time
        
        # Создание 3D графиков
        fig = plt.figure(figsize=(16, 8))
        
        # График 1: Поверхность ошибки
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, alpha=0.8)
        
        ax1.set_xlabel('Размер окна', fontsize=11)
        ax1.set_ylabel('Размер скрытого слоя', fontsize=11)
        ax1.set_zlabel('MSE', fontsize=11)
        ax1.set_title('Поверхность ошибки', fontsize=13, fontweight='bold')
        
        # Добавляем цветовую шкалу
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
        
        # Находим точку с минимальной ошибкой
        min_z_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
        min_window = window_sizes[min_z_idx[1]]
        min_hidden = hidden_sizes[min_z_idx[0]]
        min_error = Z[min_z_idx]
        
        # Отмечаем точку минимума
        ax1.scatter(min_window, min_hidden, min_error, 
                   color='red', s=100, marker='*', label=f'Min: {min_error:.4f}')
        ax1.legend()
        
        # График 2: Контурный график ошибки
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.coolwarm)
        ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
        
        ax2.set_xlabel('Размер окна', fontsize=11)
        ax2.set_ylabel('Размер скрытого слоя', fontsize=11)
        ax2.set_title('Контурный график ошибки', fontsize=13, fontweight='bold')
        
        # Отмечаем точку минимума
        ax2.scatter(min_window, min_hidden, color='red', s=100, marker='*')
        ax2.annotate(f'Оптимум\nОкно={min_window}\nСкрытый={min_hidden}\nMSE={min_error:.4f}',
                    xy=(min_window, min_hidden),
                    xytext=(min_window + 1, min_hidden + 2),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=9)
        
        # Добавляем цветовую шкалу
        fig.colorbar(contour, ax=ax2, shrink=0.8)
        
        # Добавляем общую информацию
        plt.suptitle(f'{title}\nЭпох: {epochs}, Последовательность: {len(sequence)} элементов',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Сохраняем результаты
        self.results['3d_parameter_surface'] = {
            'window_sizes': window_sizes,
            'hidden_sizes': hidden_sizes,
            'error_surface': Z,
            'training_times': training_times,
            'optimal_point': (min_window, min_hidden, min_error)
        }
    
    def plot_predictions_vs_actual(self, network, sequence: List[float],
                                  train_ratio: float = 0.7,
                                  steps_ahead: int = 3,
                                  title="Прогноз vs Фактические значения",
                                  save_path=None):
        """
        График сравнения прогнозов сети с фактическими значениями
        
        Args:
            network: Обученная нейронная сеть
            sequence: Исходная последовательность
            train_ratio: Доля данных для обучения
            steps_ahead: Количество шагов прогнозирования вперед
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        # Нормализация последовательности
        seq_min = min(sequence)
        seq_max = max(sequence)
        normalized_seq = [(x - seq_min) / (seq_max - seq_min) for x in sequence]
        
        # Разделение на обучающую и тестовую выборки
        split_idx = int(len(normalized_seq) * train_ratio)
        train_seq = normalized_seq[:split_idx]
        test_seq = normalized_seq[split_idx:]
        
        # Обучаем сеть заново на обучающей выборке (или используем переданную)
        if len(train_seq) > network.window_size + steps_ahead:
            # Создаем копию сети для чистого эксперимента
            test_network = self.network_class(
                window_size=network.window_size,
                hidden_size=network.hidden_size,
                output_size=network.output_size,
                context_reset=network.context_reset,
                elu_alpha=network.elu_alpha
            )
            
            test_network.train(train_seq, epochs=300, learning_rate=0.01)
            network_to_use = test_network
        else:
            network_to_use = network
        
        # Прогнозируем на тестовой выборке
        predictions = []
        actuals = []
        
        # Используем скользящее окно для прогнозирования
        for i in range(network_to_use.window_size, len(test_seq) - steps_ahead + 1):
            # Входное окно
            input_window = test_seq[i - network_to_use.window_size:i]
            
            # Прогнозируем несколько шагов вперед
            predicted = network_to_use.predict(input_window, steps=steps_ahead)
            
            # Фактические значения
            actual = test_seq[i:i + steps_ahead]
            
            predictions.extend(predicted)
            actuals.extend(actual)
        
        # Денормализация
        predictions_denorm = [p * (seq_max - seq_min) + seq_min for p in predictions]
        actuals_denorm = [a * (seq_max - seq_min) + seq_min for a in actuals]
        
        # Построение графика
        plt.figure(figsize=(14, 8))
        
        # Индексы для отображения
        indices = list(range(len(predictions_denorm)))
        
        # График прогнозов и фактических значений
        plt.plot(indices, actuals_denorm, 'bo-', linewidth=2, markersize=6, 
                label='Фактические значения', alpha=0.7)
        plt.plot(indices, predictions_denorm, 'ro-', linewidth=2, markersize=6,
                label='Прогнозы сети', alpha=0.7)
        
        # Добавляем линии ошибок
        for i in range(len(predictions_denorm)):
            plt.plot([i, i], [actuals_denorm[i], predictions_denorm[i]], 
                    'k--', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Шаг прогнозирования', fontsize=12)
        plt.ylabel('Значение', fontsize=12)
        plt.title(f'{title}\n(окно: {network.window_size}, скрытый: {network.hidden_size})',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Вычисляем метрики качества
        mse = np.mean((np.array(predictions_denorm) - np.array(actuals_denorm)) ** 2)
        mae = np.mean(np.abs(np.array(predictions_denorm) - np.array(actuals_denorm)))
        
        # Добавляем информацию о метриках
        metrics_text = f"MSE: {mse:.6f}\nMAE: {mae:.6f}\n"
        metrics_text += f"Точно прогнозов: {len(predictions)}\n"
        metrics_text += f"Шагов вперед: {steps_ahead}"
        
        plt.text(0.02, 0.98, metrics_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Сохраняем результаты
        self.results['predictions_vs_actual'] = {
            'predictions': predictions_denorm,
            'actuals': actuals_denorm,
            'mse': mse,
            'mae': mae
        }
    
    def plot_context_evolution(self, network, sequence: List[float],
                              steps: int = 10,
                              title="Эволюция контекстных нейронов",
                              save_path=None):
        """
        График эволюции значений контекстных нейронов во времени
        
        Args:
            network: Нейронная сеть
            sequence: Исходная последовательность
            steps: Количество шагов для анализа
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        # Нормализация последовательности
        seq_min = min(sequence)
        seq_max = max(sequence)
        normalized_seq = [(x - seq_min) / (seq_max - seq_min) for x in sequence]
        
        # Сбрасываем контекст
        network.reset_context()
        
        # Собираем историю контекстных нейронов
        elman_history = []
        jordan_history = []
        
        # Выполняем несколько шагов прямого прохода
        for i in range(min(steps, len(normalized_seq) - network.window_size)):
            # Входное окно
            input_window = normalized_seq[i:i + network.window_size]
            
            # Прямой проход (значения контекста будут обновлены)
            _ = network.forward(np.array(input_window))
            
            # Сохраняем значения контекстных нейронов
            elman_history.append(network.context_elman.flatten().copy())
            jordan_history.append(network.context_jordan.flatten().copy())
        
        # Преобразуем в массивы для удобства
        elman_history = np.array(elman_history)
        jordan_history = np.array(jordan_history)
        
        # Построение графиков
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # График 1: Контекст Элмана (скрытый слой)
        for neuron_idx in range(elman_history.shape[1]):
            ax1.plot(range(len(elman_history)), elman_history[:, neuron_idx],
                    linewidth=2, label=f'Нейрон {neuron_idx+1}')
        
        ax1.set_xlabel('Шаг времени', fontsize=12)
        ax1.set_ylabel('Значение нейрона', fontsize=12)
        ax1.set_title('Эволюция контекста Элмана (скрытый слой)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        
        # График 2: Контекст Джордана (выходной слой)
        for neuron_idx in range(jordan_history.shape[1]):
            ax2.plot(range(len(jordan_history)), jordan_history[:, neuron_idx],
                    linewidth=2, label=f'Выход {neuron_idx+1}')
        
        ax2.set_xlabel('Шаг времени', fontsize=12)
        ax2.set_ylabel('Значение выхода', fontsize=12)
        ax2.set_title('Эволюция контекста Джордана (выходной слой)',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Сохраняем результаты
        self.results['context_evolution'] = {
            'elman_history': elman_history,
            'jordan_history': jordan_history
        }
    
    def plot_comparative_analysis(self, sequence: List[float],
                                parameter_sets: List[Dict],
                                epochs: int = 300,
                                learning_rate: float = 0.01,
                                title="Сравнительный анализ различных конфигураций сети",
                                save_path=None):
        """
        Сравнительный анализ различных конфигураций сети
        
        Args:
            sequence: Исходная последовательность
            parameter_sets: Список словарей с параметрами сети
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        # Нормализация последовательности
        seq_min = min(sequence)
        seq_max = max(sequence)
        normalized_seq = [(x - seq_min) / (seq_max - seq_min) for x in sequence]
        
        results = []
        
        for i, params in enumerate(parameter_sets):
            print(f"Тестирование конфигурации {i+1}/{len(parameter_sets)}: {params}")
            
            # Создание сети с заданными параметрами
            network = self.network_class(
                window_size=params.get('window_size', 3),
                hidden_size=params.get('hidden_size', 5),
                output_size=params.get('output_size', 1),
                context_reset=params.get('context_reset', False),
                elu_alpha=params.get('elu_alpha', 1.0)
            )
            
            # Обучение сети
            start_time = time.time()
            network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
            training_time = time.time() - start_time
            
            # Оценка качества
            test_loss = network.evaluate(normalized_seq)
            
            # Сохраняем результаты
            results.append({
                'params': params,
                'final_loss': test_loss,
                'training_time': training_time,
                'loss_history': network.loss_history.copy(),
                'network': network
            })
        
        # Построение сравнительных графиков
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Финальные ошибки для каждой конфигурации
        ax1 = axes[0, 0]
        config_names = [f"Конф.{i+1}" for i in range(len(results))]
        final_losses = [r['final_loss'] for r in results]
        
        bars = ax1.bar(config_names, final_losses, color=cm.viridis(np.linspace(0, 1, len(results))))
        ax1.set_xlabel('Конфигурация сети', fontsize=12)
        ax1.set_ylabel('Финальная MSE', fontsize=12)
        ax1.set_title('Финальная ошибка для различных конфигураций', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(final_losses)*0.01,
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
        
        # График 2: Время обучения
        ax2 = axes[0, 1]
        training_times = [r['training_time'] for r in results]
        
        bars = ax2.bar(config_names, training_times, color=cm.plasma(np.linspace(0, 1, len(results))))
        ax2.set_xlabel('Конфигурация сети', fontsize=12)
        ax2.set_ylabel('Время обучения (сек)', fontsize=12)
        ax2.set_title('Время обучения для различных конфигураций',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # График 3: История обучения (сходимость)
        ax3 = axes[1, 0]
        for i, result in enumerate(results):
            ax3.plot(result['loss_history'], 
                    label=f"Конф.{i+1}", 
                    linewidth=2,
                    alpha=0.7)
        
        ax3.set_xlabel('Эпоха', fontsize=12)
        ax3.set_ylabel('MSE', fontsize=12)
        ax3.set_title('Сравнение истории обучения', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_yscale('log')
        
        # График 4: Таблица с параметрами конфигураций
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Создаем таблицу с параметрами
        table_data = []
        headers = ['Конф.', 'Окно', 'Скрытый', 'Выход', 'Сброс', 'α', 'MSE', 'Время']
        
        for i, result in enumerate(results):
            params = result['params']
            row = [
                i+1,
                params.get('window_size', '-'),
                params.get('hidden_size', '-'),
                params.get('output_size', '-'),
                'Да' if params.get('context_reset', False) else 'Нет',
                params.get('elu_alpha', '-'),
                f"{result['final_loss']:.4f}",
                f"{result['training_time']:.2f}с"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         colColours=['#f0f0f0']*len(headers))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Сохраняем результаты
        self.results['comparative_analysis'] = results
        
        # Возвращаем лучшую конфигурацию
        best_idx = np.argmin(final_losses)
        print(f"\nЛучшая конфигурация: Конф.{best_idx+1}")
        print(f"Параметры: {parameter_sets[best_idx]}")
        print(f"MSE: {final_losses[best_idx]:.6f}")
        
        return results[best_idx]['network']
    
    def save_results(self, filename: str = "visualization_results.npz"):
        """
        Сохранение результатов визуализации в файл
        
        Args:
            filename: Имя файла для сохранения
        """
        np.savez_compressed(filename, **self.results)
        print(f"Результаты сохранены в файл: {filename}")
    
    def load_results(self, filename: str = "visualization_results.npz"):
        """
        Загрузка результатов визуализации из файла
        
        Args:
            filename: Имя файла для загрузки
        """
        loaded_data = np.load(filename, allow_pickle=True)
        self.results = dict(loaded_data)
        print(f"Результаты загружены из файла: {filename}")

# Пример использования
def example_usage():
    """Пример использования визуализатора"""
    
    # Импортируем класс сети
    from neural_network import JordanElmanNetwork
    
    # Создаем визуализатор
    visualizer = NetworkVisualizer(JordanElmanNetwork)
    
    # Генерируем тестовую последовательность
    test_sequence = [1/(2**i) for i in range(15)]  # Геометрическая прогрессия
    
    # Создаем и обучаем сеть для демонстрации
    network = JordanElmanNetwork(
        window_size=3,
        hidden_size=5,
        output_size=1,
        context_reset=False,
        elu_alpha=1.0
    )
    
    # Нормализуем последовательность
    seq_min = min(test_sequence)
    seq_max = max(test_sequence)
    normalized_seq = [(x - seq_min)/(seq_max - seq_min) for x in test_sequence]
    
    # Обучаем сеть
    network.train(normalized_seq, epochs=500, learning_rate=0.01)
    
    # 1. График истории обучения
    visualizer.plot_training_history(network, 
                                    title="График зависимости количества итераций обучения от среднеквадратичной ошибки")
    
    # 2. Влияние размера окна
    visualizer.plot_window_size_impact(
        test_sequence,
        hidden_size=5,
        window_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        epochs=200,
        title="График зависимости среднеквадратичной ошибки обучения от размера скользящего окна"
    )
    
    # 3. Влияние размера скрытого слоя
    visualizer.plot_hidden_size_impact(
        test_sequence,
        window_size=3,
        hidden_sizes=[1, 2, 3, 5, 8, 10],
        epochs=200,
        title="График зависимости размера скрытого слоя от качества прогнозирования"
    )
    
    
    # 6. Сравнительный анализ разных конфигураций
    parameter_sets = [
        {'window_size': 2, 'hidden_size': 3, 'elu_alpha': 0.5},
        {'window_size': 3, 'hidden_size': 5, 'elu_alpha': 1.0},
        {'window_size': 4, 'hidden_size': 8, 'elu_alpha': 1.0},
        {'window_size': 3, 'hidden_size': 10, 'elu_alpha': 0.1},
        {'window_size': 2, 'hidden_size': 5, 'elu_alpha': 1.0, 'context_reset': True}
    ]
    
    best_network = visualizer.plot_comparative_analysis(
        test_sequence,
        parameter_sets,
        epochs=200,
        title="Сравнение различных конфигураций сети Джордана-Элмана"
    )
    
    # 7. 3D поверхность ошибки (требует больше вычислений)
    if len(test_sequence) > 10:
        visualizer.plot_3d_parameter_surface(
            test_sequence,
            window_sizes=[1, 2, 3, 4],
            hidden_sizes=[2, 3, 5, 8],
            epochs=150,
            title="Поверхность ошибки в зависимости от параметров сети"
        )
    
    # Сохраняем все результаты
    visualizer.save_results("network_visualization_results.npz")

if __name__ == "__main__":
    example_usage()