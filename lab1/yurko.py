import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm

EPOCHS = 100
p = 40

class NeuralNetwork:
    def __init__(self):
        self.W_f = None
        self.W_b = None

    def train(self, X: np.ndarray):
        ...

    def _adapt_learning_rates(self, grad_W_f, grad_W_b):
        ...

    def compress(self, X):
        return X * self.W_f

    def decompress(self, Y):
        return Y * self.W_b


BLOCK_HEIGHT = 8
BLOCK_WIDTH = 8

class ImageCompressor:
    def __init__(self, block_size=(BLOCK_HEIGHT, BLOCK_WIDTH)):
        self.block_size = block_size
        self.network = None
        self.shape = None

    def _preprocess_image(self, image_path):
        img = Image.open(image_path)
        self.shape = img.size[::-1]

        img_array = np.array(img)

        norm = (2 * img_array / 255.0) - 1
        return norm

    def _split_into_blocks(self, img):
        height, width = img.shape[:2]
        blocks = []
        num_channels = img.shape[2]

        for i in range(0, height, self.block_size[0]):
            for j in range(0, width, self.block_size[1]):
                block = np.zeros((num_channels, self.block_size[0], self.block_size[1]))

                for q in range(num_channels):
                    for x in range(self.block_size[0]):
                        for y in range(self.block_size[1]):
                            if i + x < height and j + y < width:
                                block[q, x, y] = img[i + x, j + y, q]
                blocks.append(block)
        return np.array(blocks)

    def _assemble_from_blocks(self, blocks, original_shape):

        height, width, num_channels = original_shape
        block_height, block_width = self.block_size[0], self.block_size[1]

        # Создаем пустое изображение для сборки
        reconstructed_img = np.zeros((height, width, num_channels))

        block_index = 0

        # Проходим в том же порядке, что и при разбиении
        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                # Берем следующий блок
                block = blocks[block_index]

                # Определяем реальные границы для вставки (на случай неполных блоков по краям)
                end_i = min(i + block_height, height)
                end_j = min(j + block_width, width)
                actual_block_height = end_i - i
                actual_block_width = end_j - j

                # Вставляем блок в изображение
                for q in range(num_channels):
                    for x in range(actual_block_height):
                        for y in range(actual_block_width):
                            reconstructed_img[i + x, j + y, q] = block[q, x, y]

                block_index += 1

        return reconstructed_img

    def compress(self, image_path):
        image = self._preprocess_image(image_path)

        blocks = self._split_into_blocks(image)

        self.network = NeuralNetwork()

        losses = self.network.train(blocks)

        compressed_blocks = self.network.compress(blocks)

        compress_data = {
            'compressed_blocks': compressed_blocks,
            'original_shape': self.shape,
            'block_size': self.block_size,
            'W_b': self.network.W_b,
            'W_f': self.network.W_f,
        }

        return compress_data, losses

    def decompress(self, compress_data):
        compressed_blocks = compress_data['compressed_blocks']
        original_shape = compress_data['original_shape']
        w_b = compress_data['W_b']

        reconstructed_blocks = self.network.decompress(compressed_blocks)

        reconstructed_img = self._assemble_from_blocks(reconstructed_blocks, original_shape)

        reconstructed_image = ((reconstructed_img + 1) * 255).clip(0, 255).astype(np.uint8)

        return reconstructed_image


compressor = ImageCompressor(
    block_size=(BLOCK_HEIGHT, BLOCK_WIDTH)
)

image_path = "pic/chessboard.bmp"

compressed_data, losses = compressor.compress(
    image_path
)

reconstructed_image = compressor.decompress(compressed_data)
original_image = np.array(Image.open(image_path))

plt.imshow(original_image)
plt.imshow(reconstructed_image)