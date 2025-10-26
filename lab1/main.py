import sys

import numpy as np

import cv2

from compressor import Compressor
from utils import denormalize_image
from cfg import BLOCK_WIDTH, BLOCK_HEIGHT


def get_compressed_size(compressed_data):
    total_size = 0

    if 'compressed_blocks' in compressed_data:
        total_size += compressed_data['compressed_blocks'].nbytes

    if 'W_b' in compressed_data:
        total_size += compressed_data['W_b'].nbytes

    if 'W_f' in compressed_data:
        total_size += compressed_data['W_f'].nbytes

    total_size += sys.getsizeof(compressed_data) * 2

    return total_size

compressor = Compressor(
    block_size=(BLOCK_HEIGHT, BLOCK_WIDTH)
)

image_path = "pic/256.bmp"

compressor.shape = cv2.imread(image_path).shape[:2]

compressed_data = compressor.compress(
    image_path
)

reconstructed_image = compressor.decompress(compressed_data)

final_image = denormalize_image(reconstructed_image)

original_image = cv2.imread(image_path)

combined_horizontal = np.hstack((original_image, final_image))

compressed_data = compressor.compress(image_path)

original_size = original_image.nbytes
compressed_size = get_compressed_size(compressed_data)
actual_compression_ratio = original_size / compressed_size

print(f"Размер исходного изображения: {original_size} байт")
print(f"Размер сжатых данных: {compressed_size} байт")
print(f"Коэффициент сжатия: {actual_compression_ratio:.2f}:1")

cv2.imshow('Original and reconstructed image', combined_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()