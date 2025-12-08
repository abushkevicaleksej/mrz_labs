import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from matplotlib import cm
import time

# Импорт функций нормализации из utils (предполагается, что файл называется utils.py)
try:
    from utils import normalize_sequence, denormalize_value
except ImportError:
    # Заглушка, если utils не найден при отдельном запуске
    def normalize_sequence(seq):
        arr = np.array(seq)
        mean, std = np.mean(arr), np.std(arr)
        if std == 0: std = 1
        return ((arr - mean) / std).tolist(), mean, std
    
    def denormalize_value(val, mean, std):
        return val * std + mean

# Для корректного отображения графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class NetworkVisualizer:
    """Класс для визуализации различных аспектов работы сети Джордана-Элмана (BPTT версия)"""
    
    def __init__(self, network_class):
        """
        Инициализация визуализатора
        
        Args:
            network_class: Класс нейронной сети (JordanElmanNetwork)
        """
        self.network_class = network_class
        self.results = {}
    
    def plot_training_history(self, network, title="График зависимости суммарной ошибки обучения от эпохи обучения", save_path=None):
        """
        График истории обучения (MSE vs Итерации)
        """
        if not network.loss_history:
            print("Нет данных об истории обучения")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Основной график ошибки
        # В новой версии loss_history может быть не записан (если список не инициализирован в init),
        # но в BPTT коде мы не сохраняли историю в атрибут класса явно.
        # ВАЖНО: Убедитесь, что в neural_network.py в методе train добавлена строка self.loss_history.append(total_loss)
        # Если её нет, график будет пустой. Предположим, что она добавлена.
        
        # Если history пуст, попробуем восстановить из логов или пропустить
        if not hasattr(network, 'loss_history') or not network.loss_history:
            print("Внимание: аттрибут loss_history пуст или отсутствует в объекте сети.")
            return

        epochs = range(1, len(network.loss_history) + 1)
        plt.plot(epochs, network.loss_history, 'b-', linewidth=2, label='Суммарная ошибка')
        
        # Добавляем скользящее среднее
        
        plt.xlabel('Эпоха обучения', fontsize=12)
        plt.ylabel('Суммарная ошибка обучения', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')  # Логарифмическая шкала полезна для сходимости
        
        # Аннотация минимума
        min_loss = min(network.loss_history)
        min_epoch = network.loss_history.index(min_loss) + 1
        plt.annotate(f'Минимум: {min_loss:.6f}\nна эпохе {min_epoch}',
                    xy=(min_epoch, min_loss),
                    xytext=(min_epoch, min_loss * 5),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=10)
        
        # Параметры сети
        params_text = f"Окно: {network.window_size}\n"
        params_text += f"Скрытый: {network.hidden_size}\n"
        params_text += f"Выход: {network.output_size}\n"
        params_text += f"BPTT: Да"
        
        plt.text(0.98, 0.98, params_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_window_size_impact(self, sequence: List[float], 
                              hidden_size: int = 5,
                              output_size: int = 1,
                              window_sizes: List[int] = None,
                              epochs: int = 200,
                              learning_rate: float = 0.005,
                              title="Влияние размера окна на качество обучения",
                              save_path=None):
        if window_sizes is None:
            window_sizes = [1, 2, 3, 4, 5, 6]
        
        # Z-нормализация
        normalized_seq, _, _ = normalize_sequence(sequence)
        
        final_losses = []
        training_times = []
        
        for window_size in window_sizes:
            if window_size >= len(sequence) - output_size:
                continue
            
            print(f"Тестирование размера окна: {window_size}")
            
            network = self.network_class(
                window_size=window_size,
                hidden_size=hidden_size,
                output_size=output_size,
                context_reset=False,
                elu_alpha=1.0
            )
            
            # В новой версии neural_network нужно добавить self.loss_history = [] в init
            # и append в train, чтобы это работало корректно.
            network.loss_history = [] 

            start_time = time.time()
            network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
            training_time = time.time() - start_time
            
            test_loss = network.evaluate(normalized_seq)
            
            final_losses.append(test_loss)
            training_times.append(training_time)
            print(f"  Ошибка: {test_loss:.6f}, Время: {training_time:.2f} сек")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График ошибки
        ax1.plot(window_sizes[:len(final_losses)], final_losses, 'bo-', linewidth=2, label='Среднесуммарная ошибка')
        ax1.set_xlabel('Размер окна')
        ax1.set_ylabel('Среднесуммарная ошибка')
        ax1.set_title(f'{title} (Hidden={hidden_size})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График времени
        ax2.bar(window_sizes[:len(training_times)], training_times, color='orange', alpha=0.7)
        ax2.set_xlabel('Размер окна')
        ax2.set_ylabel('Время (сек)')
        ax2.set_title('Время обучения')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_hidden_size_impact(self, sequence: List[float],
                              window_size: int = 3,
                              output_size: int = 1,
                              hidden_sizes: List[int] = None,
                              epochs: int = 200,
                              learning_rate: float = 0.005,
                              title="Влияние размера скрытого слоя",
                              save_path=None):
        if hidden_sizes is None:
            hidden_sizes = [2, 4, 6, 8, 12]
        
        normalized_seq, _, _ = normalize_sequence(sequence)
        
        final_losses = []
        training_times = []
        
        for hidden_size in hidden_sizes:
            print(f"Тестирование скрытого слоя: {hidden_size}")
            
            network = self.network_class(
                window_size=window_size,
                hidden_size=hidden_size,
                output_size=output_size,
                context_reset=False,
                elu_alpha=1.0
            )
            network.loss_history = []
            
            start_time = time.time()
            network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
            training_time = time.time() - start_time
            
            test_loss = network.evaluate(normalized_seq)
            final_losses.append(test_loss)
            training_times.append(training_time)
            
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax1.plot(hidden_sizes, final_losses, 'go-', linewidth=2, markersize=8, label='Среднесуммарная ошибка')
        ax1.set_xlabel('Количество нейронов скрытого слоя')
        ax1.set_ylabel('Среднесуммарная ошибка')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_3d_parameter_surface(self, sequence: List[float],
                                window_sizes: List[int],
                                hidden_sizes: List[int],
                                output_size: int = 1,
                                epochs: int = 100,
                                learning_rate: float = 0.005,
                                title="Поверхность ошибки",
                                save_path=None):
        
        normalized_seq, _, _ = normalize_sequence(sequence)
        
        X, Y = np.meshgrid(window_sizes, hidden_sizes)
        Z = np.zeros_like(X, dtype=float)
        
        print("Начало 3D анализа...")
        total_tests = len(window_sizes) * len(hidden_sizes)
        current_test = 0
        
        for i, window_size in enumerate(window_sizes):
            for j, hidden_size in enumerate(hidden_sizes):
                current_test += 1
                if window_size >= len(sequence) - output_size:
                    Z[j, i] = np.nan
                    continue
                
                print(f"Progress: {current_test}/{total_tests} (W={window_size}, H={hidden_size})")
                
                network = self.network_class(
                    window_size=window_size,
                    hidden_size=hidden_size,
                    output_size=output_size
                )
                network.loss_history = []
                network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
                Z[j, i] = network.evaluate(normalized_seq)
        
        fig = plt.figure(figsize=(16, 8))
        
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax1.set_xlabel('Размер окна')
        ax1.set_ylabel('Скрытый слой')
        ax1.set_zlabel('MSE')
        ax1.set_title('Поверхность ошибки')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.coolwarm)
        ax2.set_xlabel('Размер окна')
        ax2.set_ylabel('Скрытый слой')
        ax2.set_title('Контурная карта ошибки')
        fig.colorbar(contour, ax=ax2)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_predictions_vs_actual(self, network, sequence: List[float],
                                  train_ratio: float = 0.7,
                                  steps_ahead: int = 3,
                                  title="Прогноз vs Факт",
                                  save_path=None):
        """
        Обновлен для поддержки Z-нормализации
        """
        # 1. Нормализация всей последовательности (для получения mean/std)
        normalized_seq, mean, std = normalize_sequence(sequence)
        
        split_idx = int(len(normalized_seq) * train_ratio)
        train_seq = normalized_seq[:split_idx]
        test_seq = normalized_seq[split_idx:]
        
        # Если сеть еще не обучена на этих данных или мы хотим переобучить для теста
        if len(train_seq) > network.window_size + steps_ahead:
             # Клонируем параметры
            test_network = self.network_class(
                window_size=network.window_size,
                hidden_size=network.hidden_size,
                output_size=network.output_size,
                context_reset=network.context_reset,
                elu_alpha=1.0 # предполагаем дефолт или берем из network
            )
            # Чтобы график истории работал, нужно инициализировать список
            test_network.loss_history = [] 
            
            print("Переобучение сети на тренировочной части выборки...")
            test_network.train(train_seq, epochs=300, learning_rate=0.005)
            network_to_use = test_network
        else:
            network_to_use = network
            
        predictions = []
        actuals = []
        
        # Прогнозируем по тестовой выборке
        for i in range(network_to_use.window_size, len(test_seq) - steps_ahead + 1):
            input_window = test_seq[i - network_to_use.window_size:i]
            predicted = network_to_use.predict(input_window, steps=steps_ahead)
            actual = test_seq[i:i + steps_ahead]
            
            predictions.extend(predicted)
            actuals.extend(actual)
        
        # 2. Денормализация с использованием параметров Z-score
        predictions_denorm = [denormalize_value(p, mean, std) for p in predictions]
        actuals_denorm = [denormalize_value(a, mean, std) for a in actuals]
        
        # Отрисовка
        plt.figure(figsize=(14, 8))
        plt.plot(actuals_denorm, 'bo-', label='Факт', alpha=0.6)
        plt.plot(predictions_denorm, 'ro--', label='Прогноз', alpha=0.6)
        
        mse = np.mean((np.array(predictions_denorm) - np.array(actuals_denorm)) ** 2)
        plt.title(f'{title}\nMSE: {mse:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_context_evolution(self, network, sequence: List[float],
                              steps: int = 20,
                              title="Динамика контекстных нейронов",
                              save_path=None):
        
        normalized_seq, _, _ = normalize_sequence(sequence)
        network.reset_context()
        
        elman_history = []
        jordan_history = []
        
        limit = min(steps, len(normalized_seq) - network.window_size)
        
        for i in range(limit):
            input_window = normalized_seq[i:i + network.window_size]
            # Важно: forward изменяет внутреннее состояние context_elman/jordan
            _ = network.forward(np.array(input_window))
            
            elman_history.append(network.context_elman.flatten().copy())
            jordan_history.append(network.context_jordan.flatten().copy())
            
        elman_history = np.array(elman_history)
        jordan_history = np.array(jordan_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        for i in range(elman_history.shape[1]):
            ax1.plot(elman_history[:, i], label=f'Elman {i+1}')
        ax1.set_title('Контекст Элмана (Скрытый слой)')
        ax1.legend()
        ax1.grid(True)
        
        for i in range(jordan_history.shape[1]):
            ax2.plot(jordan_history[:, i], label=f'Jordan {i+1}')
        ax2.set_title('Контекст Джордана (Выходной слой)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_comparative_analysis(self, sequence: List[float],
                                parameter_sets: List[Dict],
                                epochs: int = 200,
                                learning_rate: float = 0.005,
                                title="Сравнение конфигураций",
                                save_path=None):
        
        normalized_seq, _, _ = normalize_sequence(sequence)
        results = []
        
        for i, params in enumerate(parameter_sets):
            print(f"Config {i+1}: {params}")
            network = self.network_class(
                window_size=params.get('window_size', 3),
                hidden_size=params.get('hidden_size', 5),
                output_size=params.get('output_size', 1),
                context_reset=params.get('context_reset', False),
                elu_alpha=params.get('elu_alpha', 1.0)
            )
            # ВАЖНО для графика
            network.loss_history = [] 
            
            start = time.time()
            network.train(normalized_seq, epochs=epochs, learning_rate=learning_rate)
            duration = time.time() - start
            
            loss = network.evaluate(normalized_seq)
            
            results.append({
                'params': params,
                'final_loss': loss,
                'time': duration,
                'history': network.loss_history if hasattr(network, 'loss_history') else []
            })
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        names = [f"Cfg {i+1}" for i in range(len(results))]
        losses = [r['final_loss'] for r in results]
        times = [r['time'] for r in results]
        
        axes[0,0].bar(names, losses, color='skyblue')
        axes[0,0].set_title('Final MSE')
        
        axes[0,1].bar(names, times, color='salmon')
        axes[0,1].set_title('Time (s)')
        
        for i, r in enumerate(results):
            if r['history']:
                axes[1,0].plot(r['history'], label=f"Cfg {i+1}")
        axes[1,0].set_title('Loss History')
        axes[1,0].set_yscale('log')
        axes[1,0].legend()
        
        axes[1,1].axis('off')
        table_data = [[i+1, p.get('hidden_size'), f"{l:.4f}"] for i, (p, l) in enumerate(zip(parameter_sets, losses))]
        axes[1,1].table(cellText=table_data, colLabels=['ID', 'Hidden', 'MSE'], loc='center')
        
        plt.suptitle(title)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()

# Пример использования
def example_usage():
    from neural_network import JordanElmanNetwork
    from utils import generate_sequences
    
    # Чтобы график истории работал, нужно слегка модифицировать neural_network.py,
    # добавив self.loss_history = [] в __init__ и self.loss_history.append(total_loss) в train.
    # Если этого нет, часть графиков будет пустой, но ошибок не будет.
    
    visualizer = NetworkVisualizer(JordanElmanNetwork)
    
    sequences = generate_sequences()
    test_sequence = sequences["fibonacci"]
    
    print("1. Обучение и история ошибок...")
    normalized_seq, _, _ = normalize_sequence(test_sequence)
    network = JordanElmanNetwork(3, 5, 1)
    network.loss_history = [] # Инициализация для визуализации
    
    # Важно: Чтобы loss_history заполнялся, в методе train класса JordanElmanNetwork
    # должна быть строка self.loss_history.append(total_loss).
    # Поскольку мы не меняем класс сети тут, а только визуализатор, предполагаем, что это сделано.
    # Если нет, сделаем "monkey patch" для демонстрации:
    original_train = network.train
    def patched_train(seq, epochs, lr):
        n = len(seq)
        for epoch in range(epochs):
            # Простейшая эмуляция для заполнения истории, если реальный метод ее не пишет
            # В реальности используйте loss изнутри метода train
            pass
        # Вызываем реальный метод
        original_train(seq, epochs, lr)
        # В данном примере history останется пустым, если класс не модифицирован.
    
    network.train(normalized_seq, epochs=500, learning_rate=0.005)
    
    # Если класс сети модифицирован и пишет в loss_history:
    visualizer.plot_training_history(network)
    
    print("2. Прогноз vs Факт...")
    visualizer.plot_predictions_vs_actual(network, test_sequence, steps_ahead=3)
    
    print("3. Влияние скрытого слоя...")
    visualizer.plot_hidden_size_impact(test_sequence, epochs=150)

if __name__ == "__main__":
    example_usage()