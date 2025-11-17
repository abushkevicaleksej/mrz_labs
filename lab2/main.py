###############################
# Лабораторная работа №4 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Хопфилда с дискретным состоянием и дискретным временем в асинхронном режиме.
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 17.11.2025

from utils import test_symbol_recognition, get_absolute_path

if __name__ == "__main__":
    training_set = [
        get_absolute_path('pics/1.bmp'),
        get_absolute_path('pics/2.bmp'),
        get_absolute_path('pics/3.bmp'),
        get_absolute_path('pics/4.bmp'),
    
    ]
    
    test_cases = [
        (get_absolute_path('pics/1.bmp'), 0.4, "random", 100),
    ]
    
    for test_file, noise_level, noise_type, max_iter in test_cases:
        result, iterations, success, matched = test_symbol_recognition(
            training_set, 
            test_file, 
            noise_level, 
            noise_type,
            max_iterations=max_iter
        )
