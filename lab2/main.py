from utils import test_symbol_recognition

if __name__ == "__main__":
    training_symbols = ['A', 'B', 'pics/dog.jpg']
    
    test_cases = [
        ('A', 0.1, "random"),
        ('B', 0.1, "random"),
        ('pics/dog.jpg', 0.2, "both"),
    ]
    
    for test_symbol, noise_level, noise_type in test_cases:
        if test_symbol in training_symbols:
            result, iterations = test_symbol_recognition(
                training_symbols, 
                test_symbol, 
                noise_level, 
                noise_type,
                image_size=(20, 20) 
            )

    # TODO получение количества итераций до достижения состояния релаксации