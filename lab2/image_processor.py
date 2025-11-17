###############################
# Лабораторная работа №4 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Хопфилда с дискретным состоянием и дискретным временем в асинхронном режиме.
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл, содержащий реализацию процессора изображений
# Дата 17.11.2025

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ImageProcessor:
    def __init__(self, image_size=None):
        self.image_size = image_size
        
    def load_image(self, file_path):
        try:
            img = Image.open(file_path).convert('L')
            if self.image_size:
                img = img.resize(self.image_size)
            else:
                self.image_size = img.size
            return img
        except Exception as e:
            print(f"Ошибка загрузки изображения {file_path}: {e}")
            return None

    def image_to_vector(self, image):
        img_array = np.array(image)
        vector = np.where(img_array < 128, 1, -1)
        return vector.flatten()
    
    def vector_to_image(self, vector):
        if not self.image_size:
            raise ValueError("Размер изображения не определен")
        
        matrix = vector.reshape(self.image_size[1], self.image_size[0])
        img_array = np.where(matrix == 1, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='L')
    
    def add_noise(self, vector, noise_level=0.3, noise_type="both"):
        noisy_vector = vector.copy()
        n_pixels = len(vector)
        
        noise_indices = np.random.choice(n_pixels, size=int(noise_level * n_pixels), replace=False)
        
        if noise_type == "zeros":
            noisy_vector[noise_indices] = 0
        elif noise_type == "random":
            random_values = np.random.choice([-1, 1], size=len(noise_indices))
            noisy_vector[noise_indices] = random_values
        elif noise_type == "both":
            half = len(noise_indices) // 2
            if half > 0:
                noisy_vector[noise_indices[:half]] = 0
            if len(noise_indices) - half > 0:
                random_values = np.random.choice([-1, 1], size=len(noise_indices) - half)
                noisy_vector[noise_indices[half:]] = random_values
        
        return noisy_vector
    
    def visualize_patterns(self, original, noisy, reconstructed, original_symbol="", iterations=0):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        orig_img = self.vector_to_image(original)
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title(f'Оригинал: "{original_symbol}"')
        axes[0].axis('off')
        
        noisy_display = noisy.copy()
        noisy_display[noisy == 0] = 0 
        noisy_img = self.vector_to_image(np.where(noisy_display == 0, -1, noisy_display))
        axes[1].imshow(noisy_img, cmap='gray')
        axes[1].set_title(f'Зашумленное\n(уровень шума)')
        axes[1].axis('off')
        
        recon_img = self.vector_to_image(reconstructed)
        axes[2].imshow(recon_img, cmap='gray')
        axes[2].set_title(f'Восстановленное\n{iterations} итераций')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()