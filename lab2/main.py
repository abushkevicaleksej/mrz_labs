###############################
# Лабораторная работа №4 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Хопфилда с дискретным состоянием и дискретным временем в асинхронном режиме.
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Главный файл программы
# Дата 17.11.2025

from utils import test_symbol_recognition, get_absolute_path

if __name__ == "__main__":
    training_set = [
        # get_absolute_path('pics/a.bmp'),
        # get_absolute_path('pics/b.bmp'),
        # get_absolute_path('pics/c.bmp'),
        # get_absolute_path('pics/d.bmp'),
        # get_absolute_path('pics/e.bmp'),
        # get_absolute_path('pics/f.bmp'),
        # get_absolute_path('pics/g.bmp'),
        # get_absolute_path('pics/h.bmp'),
        # get_absolute_path('pics/i.bmp'),
        # get_absolute_path('pics/j.bmp'),
        # get_absolute_path('pics/k.bmp'),
        # get_absolute_path('pics/l.bmp'),
        # get_absolute_path('pics/m.bmp'),
        # get_absolute_path('pics/n.bmp'),
        # get_absolute_path('pics/o.bmp'),
        # get_absolute_path('pics/p.bmp'),
        # get_absolute_path('pics/q.bmp'),
        # get_absolute_path('pics/r.bmp'),
        # get_absolute_path('pics/s.bmp'),
        # get_absolute_path('pics/t.bmp'),
        # get_absolute_path('pics/u.bmp'),
        # get_absolute_path('pics/v.bmp'),
        # get_absolute_path('pics/w.bmp'),
        # get_absolute_path('pics/x.bmp'),
        # get_absolute_path('pics/y.bmp'),
        # get_absolute_path('pics/z.bmp'),
        # get_absolute_path('pics/1.bmp'),
        # get_absolute_path('pics/2.bmp'),
        # get_absolute_path('pics/3.bmp'),
        # get_absolute_path('pics/4.bmp'),
        # get_absolute_path('pics/5.bmp'),
        # get_absolute_path('pics/6.bmp'),
        # get_absolute_path('pics/7.bmp'),
        # get_absolute_path('pics/8.bmp'),
        # get_absolute_path('pics/9.bmp'),
        # get_absolute_path('pics/0.bmp'),
        get_absolute_path('pics/1_test.bmp'),
        get_absolute_path('pics/2_test.bmp'),
        get_absolute_path('pics/3_test.bmp'),
        get_absolute_path('pics/4_test.bmp'),
        get_absolute_path('pics/5_test.bmp'),
        get_absolute_path('pics/6_test.bmp'),
        get_absolute_path('pics/7_test.bmp'),
        get_absolute_path('pics/8_test.bmp'),
    ]
    
    test_cases = [
        (get_absolute_path('pics/8_test.bmp'), 0.0, "random", 100),
    ]
    
    for test_file, noise_level, noise_type, max_iter in test_cases:
        result, iterations, success, matched = test_symbol_recognition(
            training_set, 
            test_file, 
            noise_level, 
            noise_type,
            max_iterations=max_iter
        )
