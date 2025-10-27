###############################
# Лабораторная работа №3 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами.
# Выполнил студент группы 221701 БГУИР Абушкевич Алексей Александрович
# Файл, содержащий реализацию загрузчика изображений
# Дата 25.10.2025

from typing import Iterator

import cv2
import numpy as np

from utils import normalize_image, split_into_blocks, flatten

class Loader:
    def __init__(self, image_path: str, r: int, m: int) -> None:
        self.items = []

        image = cv2.imread(image_path)

        image = normalize_image(image)
        rect = split_into_blocks(image, r, m)
        vectors = flatten(rect)

        for vec in vectors:
            self.items.append(vec)

    def __getitem__(self, idx) -> np.array:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[np.array]:
        return iter(self.items)