import numpy as np
import matplotlib.pyplot as plt
from neural_network import JordanElmanNetwork
from utils import generate_sequences, zscore_normalize
import plotter

SEQUENCE_NAME = "harmonic"
raw_sequence = generate_sequences()[SEQUENCE_NAME]
INPUT_RAW = raw_sequence[:15]
processed_input, mean, std = zscore_normalize(INPUT_RAW)

BASE_PARAMS = {
    'seq': processed_input,
    'input_size': 4,
    'hidden_size': 5,
    'context_size': 5,
    'effector_size': 1,
    'alpha': 0.001,
    'max_errors': 1e-6,
    'max_iters': 10000,
    'predict_len': 1,
    'reset_context': True,
    'effector_activation_type': 'elu',
    'verbose': False
}

def run_experiment_1_error_vs_epoch():
    print("--- Построение графика: Ошибка от Эпохи ---")
    net = JordanElmanNetwork(**BASE_PARAMS)
    iters, history = net.train()
    
    # Строим график
    plotter.plot_dependency(
        x_data=range(len(history)),
        y_data=history,
        x_label="Итерация обучения",
        y_label="Величина ошибки",
        title="График зависимости количества итераций обучения от величины ошибки",
    )

def run_experiment_2_error_vs_window_size():
    print("--- Построение графика: Ошибка от Размера Окна ---")
    window_sizes = [2, 3, 4, 5, 6, 8, 10, 12, 14]
    final_errors = []

    for w in window_sizes:
        print(f"Тестируем размер окна: {w}")
        # Копируем параметры и меняем input_size
        params = BASE_PARAMS.copy()
        params['input_size'] = w
        # Скрытый слой должен быть >= context, увеличим если надо
        params['hidden_size'] = max(params['hidden_size'], params['context_size']) 
        
        net = JordanElmanNetwork(**params)
        _, history = net.train()
        final_errors.append(history[-1])

    plotter.plot_dependency(
        x_data=window_sizes,
        y_data=final_errors,
        x_label="Размер скользящего окна",
        y_label="Величина ошибки",
        title="График зависимости величины ошибки от размера скользящего окна"
    )

def run_experiment_3_error_vs_hidden_size():
    print("--- Построение графика: Ошибка от Скрытого Слоя ---")
    hidden_sizes = [6, 8, 10, 15]
    final_errors = []

    for h in hidden_sizes:
        print(f"Тестируем размер скрытого слоя: {h}")
        params = BASE_PARAMS.copy()
        params['hidden_size'] = h
        # Context не может быть больше hidden
        params['context_size'] = min(params['context_size'], h)
        
        net = JordanElmanNetwork(**params)
        _, history = net.train()
        final_errors.append(history[-1])

    plotter.plot_dependency(
        x_data=hidden_sizes,
        y_data=final_errors,
        x_label="Размер скрытого слоя",
        y_label="Величина ошибки",
        title="График зависимости величины ошибки от количества нейронов скрытого слоя"
    )

def run_experiment_4_error_vs_alpha():
    print("--- Построение графика: Ошибка от Скорости обучения (Alpha) ---")
    
    # Список скоростей обучения (логарифмическая шкала)
    alphas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    final_errors = []

    for lr in alphas:
        print(f"Тестируем скорость обучения: {lr}")
        params = BASE_PARAMS.copy()
        params['alpha'] = lr
        # Увеличим число итераций, так как малая скорость требует больше времени
        params['max_iters'] = 3000 
        
        # Фиксируем seed, чтобы влияние случайной инициализации весов было одинаковым
        np.random.seed(42) 
        
        net = JordanElmanNetwork(**params)
        _, history = net.train()
        
        # Берем среднюю ошибку за последние 10 эпох для стабильности
        final_errors.append(np.mean(history[-10:]))

    plotter.plot_dependency(
        x_data=alphas,
        y_data=final_errors,
        x_label="Коэффициент обучения",
        y_label="Величина ошибки",
        title="График зависимости величины ошибки от коэффициента обучения",
        x_log_scale=True,  # Важно для Alpha
        y_log_scale=True   # Ошибка тоже может сильно меняться
    )

def run_experiment_5_error_vs_context_size():
    print("--- Построение графика: Ошибка от Размера Контекста ---")
    
    # Чтобы проверить большие контексты, нужен большой скрытый слой
    fixed_hidden_size = 10
    
    # Размер контекста от 1 до размера скрытого слоя
    context_sizes = list(range(1, fixed_hidden_size + 1))
    final_errors = []

    for ctx in context_sizes:
        print(f"Тестируем размер контекста: {ctx}")
        params = BASE_PARAMS.copy()
        params['hidden_size'] = fixed_hidden_size
        params['context_size'] = ctx
        
        np.random.seed(42) # Для воспроизводимости
        
        net = JordanElmanNetwork(**params)
        _, history = net.train()
        final_errors.append(history[-1])

    plotter.plot_dependency(
        x_data=context_sizes,
        y_data=final_errors,
        x_label="Размер контекстного слоя",
        y_label="Величина ошибки",
        title=f"График зависимости величины ошибки от размера контекстного слоя"
    )

if __name__ == "__main__":
    # run_experiment_1_error_vs_epoch()
    # run_experiment_2_error_vs_window_size()
    # run_experiment_3_error_vs_hidden_size()
    
    # run_experiment_4_error_vs_alpha()
    run_experiment_5_error_vs_context_size()