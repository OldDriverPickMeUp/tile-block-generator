import os
import random

from .common import CanNotGenerateError
from .core import MatrixData, MatrixT2D, Model2D, Overlap2D
from .readers import ImageReader, MatrixTReader2D


class Adapter:
    def __init__(self, tile_folder, example_data):
        self._folder = tile_folder
        self._categories = example_data['categories']
        self._example_mat = example_data["example"]
        self._image_reader = self.get_reader()

    def get_reader(self):
        tile_filenames = [os.path.join(self._folder, each["filename"]) for each in self._categories[1:]]
        return ImageReader(tile_filenames)

    def get_matrix_data(self):
        reader = MatrixTReader2D()
        reader.load_example(self._example_mat)
        rules, categories = reader.generate_rules()
        return MatrixData(rules), list(categories)

    def generate(self, size, file_prefix):
        mat_data, categories = self.get_matrix_data()
        mat_T = MatrixT2D(mat_data)
        for each in range(1000):
            random.seed()
            seed = random.randint(10000, 100000000)
            model = Model2D(size, categories, mat_T)
            try:
                loop = model.generate(seed, save_snapshot=False, reader=self._image_reader, save_prefix='out/example',
                                      max_loop=2000)
            except CanNotGenerateError as e:
                print('==', each, size, seed, 'fail', e)
                continue
            model.save_snapshot(self._image_reader, f'{file_prefix}-{size}-{seed}')
            print('==', 'find proper generated', loop, f'seed={seed}')
            break

    def generate_overlap(self, output_size, overlap_units, block_size, file_prefix):
        mat_data, categories = self.get_matrix_data()
        mat_T = MatrixT2D(mat_data)
        for each in range(1000):
            random.seed()
            seed = random.randint(10000, 100000000)
            model = Overlap2D(output_size, overlap_units, block_size, categories, mat_T)
            try:
                model.generate(seed, save_snapshot=False, reader=self._image_reader, save_prefix='out/example',
                               max_loop=300,
                               max_block_loop=100, save_block_snapshot=False)
            except CanNotGenerateError:
                print('==', each, output_size, block_size, overlap_units, seed, 'fail')
                continue
            except Exception as e:
                raise e
            model.save_snapshot(self._image_reader, f'{file_prefix}-{output_size}-{block_size}-{overlap_units}-{seed}')
            print('==', 'find proper generated')
            break
