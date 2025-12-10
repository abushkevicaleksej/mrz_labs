import matplotlib.pyplot as plt

def plot_dependency(x_data, y_data, x_label, y_label, title, y_log_scale=False, x_log_scale=False):
    """
    Универсальная функция для построения графика зависимости Y от X.
    Добавлена поддержка логарифмической шкалы для X и Y.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='r')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, which="both", ls="-") # "both" включает сетку для log scale
    
    if y_log_scale:
        plt.yscale('log')
    if x_log_scale:
        plt.xscale('log')
        
    plt.show()

def plot_comparison(real, predicted, title="Сравнение эталона и прогноза"):
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Эталон', marker='.', color='green')
    plt.plot(predicted, label='Прогноз', marker='x', linestyle='--', color='red')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()