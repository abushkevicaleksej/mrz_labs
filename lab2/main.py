from utils import test_symbol_recognition, get_absolute_path

if __name__ == "__main__":
    training_set = [
        get_absolute_path('pics/dog.jpg'),
        get_absolute_path('pics/poison.jpeg')
    
    ]
    
    test_cases = [
        (get_absolute_path('pics/dog.jpg'), 0.95, "both", 100),
    ]
    
    for test_file, noise_level, noise_type, max_iter in test_cases:
        result, iterations, success, matched = test_symbol_recognition(
            training_set, 
            test_file, 
            noise_level, 
            noise_type,
            max_iterations=max_iter
        )
