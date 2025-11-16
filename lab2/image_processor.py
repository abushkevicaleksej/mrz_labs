import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

class ImageProcessor:
    def __init__(self, image_size=(10, 10)):
        self.image_size = image_size
        self.width, self.height = image_size
        
    def create_symbol_image(self, symbol, font_size=20):

        img = Image.new('L', self.image_size, color=255)
        draw = ImageDraw.Draw(img)
        
        font = ImageFont.truetype("arial.ttf", font_size)
        
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (self.width - text_width) // 2 - bbox[0]
        y = (self.height - text_height) // 2 - bbox[1]
        
        draw.text((x, y), symbol, font=font, fill=0)
        
        return img
    
    def image_to_vector(self, image):

        img_array = np.array(image)
        
        vector = np.where(img_array < 128, 1, -1)
        
        return vector.flatten()
    
    def vector_to_image(self, vector):

        matrix = vector.reshape(self.height, self.width)
        
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
        
        noisy_img = self.vector_to_image(np.where(noisy == 0, 1, noisy))
        axes[1].imshow(noisy_img, cmap='gray')
        axes[1].set_title(f'Зашумленное (уровень шума)')
        axes[1].axis('off')
        
        recon_img = self.vector_to_image(reconstructed)
        axes[2].imshow(recon_img, cmap='gray')
        axes[2].set_title(f'Восстановленное\n{iterations} итераций')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()