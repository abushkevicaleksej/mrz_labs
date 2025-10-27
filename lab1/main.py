###############################
# Лабораторная работа №3 по дисциплине МРЗвИС
# Выполнил студент группы 221701 БГУИР Абушкевич Алексей Александрович
# Главный файл программы

import numpy as np

import cv2

from compressor import Compressor
from utils import denormalize_image, get_compressed_size
from cfg import BLOCK_WIDTH, BLOCK_HEIGHT

compressor = Compressor(
    block_size=(BLOCK_HEIGHT, BLOCK_WIDTH)
)

image_path = "D:\\Repos\\src\\mrz_labs\\lab1\\pic\\1024.bmp"

compressor.shape = cv2.imread(image_path).shape[:2]

compressed_data = compressor.compress(
    image_path
)

reconstructed_image = compressor.decompress(compressed_data)

final_image = denormalize_image(reconstructed_image)

original_image = cv2.imread(image_path)

combined_horizontal = np.hstack((original_image, final_image))

original_size = original_image.nbytes
compressed_size = get_compressed_size(compressed_data)
actual_compression_ratio = original_size / compressed_size

print(f"Размер исходного изображения: {original_size} байт")
print(f"Размер сжатых данных: {compressed_size} байт")
print(f"Коэффициент сжатия: {actual_compression_ratio:.2f}:1")

cv2.imshow('Original and reconstructed image', combined_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()