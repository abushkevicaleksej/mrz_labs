from neural_network import NeuralNetwork
from loader import Loader
from cfg import P, EPOCHS, MAX_ERROR

class Compressor:
    def __init__(self, block_size):
        self.block_size = block_size
        self.network = None
        self.shape = None

    def compress(self, image_path):
        self.network = NeuralNetwork()
        self.network.init_weights(3 * self.block_size[0] * self.block_size[1], P)

        loader = Loader(image_path, self.block_size[0], self.block_size[1])

        for epoch, error in self.network.train(loader, EPOCHS, MAX_ERROR):
            print(f"Epoch {epoch}, Error: {error:.6f}")

        compressed_blocks = self.network.compress(loader.items)

        compress_data = {
            'compressed_blocks': compressed_blocks,
            'original_shape': self.shape,
            'W_f': self.network.W_f,
            'W_b': self.network.W_b,
        }

        return compress_data

    def decompress(self, compress_data):
        from utils import assemble_from_blocks

        compressed_blocks = compress_data['compressed_blocks']
        original_shape = compress_data['original_shape']

        reconstructed_blocks = self.network.decompress(compressed_blocks)

        reconstructed_img = assemble_from_blocks(reconstructed_blocks, original_shape)

        return reconstructed_img
