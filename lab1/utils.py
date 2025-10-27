###############################
# Лабораторная работа №3 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами.
# Выполнил студент группы 221701 БГУИР Абушкевич Алексей Александрович
# Файл, содержащий реализацию вспомогательных функций
# Дата 25.10.2025

import sys

import numpy as np
from scipy.stats import normaltest, truncnorm, shapiro

from cfg import BLOCK_WIDTH, BLOCK_HEIGHT

def adaptive_a_func(arr, c=1000) -> float:
    quad_sum = np.sum(arr ** 2) + c
    return 1 / (quad_sum)

def flatten(rectangles: np.ndarray) -> np.ndarray:
    num_rectangles = rectangles.shape[0]
    flattened = rectangles.reshape(num_rectangles, -1)
    return flattened

def denormalize_image(image: np.ndarray) -> np.ndarray:
    image = (image + 1) * 255 / 2
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return 2 * image / 255 - 1

def split_into_blocks(image, r, m):
    assert image.ndim == 3
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    num_rows = int(np.ceil(h / r))
    num_cols = int(np.ceil(w / m))
    total_rectangles = num_rows * num_cols

    rectangles = np.zeros((total_rectangles, r, m, channels), dtype=image.dtype)

    rect_idx = 0
    for i in range(num_rows):
        row_start = i * r

        if row_start + r > h:
            row_start = h - r

        for j in range(num_cols):
            col_start = j * m

            if col_start + m > w:
                col_start = w - m

            rect_y_end = row_start + r
            rect_x_end = col_start + m

            rect_height = rect_y_end - row_start
            rect_width = rect_x_end - col_start

            rect = np.zeros((r, m, channels), dtype=image.dtype)
            rect[:rect_height, :rect_width, :] = image[row_start:rect_y_end, col_start:rect_x_end, :]

            rectangles[rect_idx] = rect
            rect_idx += 1

    return rectangles

def assemble_from_blocks(blocks, original_shape):
    r, m = BLOCK_WIDTH, BLOCK_HEIGHT
    h, w = original_shape
    channels = 3

    rectangles = blocks.reshape(-1, r, m, channels)

    num_rows = int(np.ceil(h / r))
    num_cols = int(np.ceil(w / m))

    reconstructed = np.zeros((h, w, channels), dtype=blocks.dtype)

    rect_idx = 0
    for i in range(num_rows):
        row_start = i * r
        if row_start + r > h:
            row_start = h - r

        for j in range(num_cols):
            col_start = j * m
            if col_start + m > w:
                col_start = w - m

            rect_y_end = min(row_start + r, h)
            rect_x_end = min(col_start + m, w)

            rect = rectangles[rect_idx]

            reconstructed[row_start:rect_y_end,
            col_start:rect_x_end, :] = rect[:rect_y_end - row_start, :rect_x_end - col_start, :]

            rect_idx += 1

    return reconstructed

def get_compressed_size(compressed_data):
    total_size = 0

    if 'compressed_blocks' in compressed_data:
        total_size += compressed_data['compressed_blocks'].nbytes

    if 'W_b' in compressed_data:
        total_size += compressed_data['W_b'].nbytes

    if 'W_f' in compressed_data:
        total_size += compressed_data['W_f'].nbytes

    total_size += sys.getsizeof(compressed_data)

    return total_size

def check_normality(arr, alpha = 0.05) -> bool:
    if len(arr) < 3:
        return True 
    
    statistic, p_value = shapiro(arr)
    return p_value > alpha

def truncated_normal(shape, mean=0, std=1, clip=3):
    weights = np.random.normal(mean, std, size=shape).astype(np.float32)
    return np.clip(weights, -clip*std, clip*std)
