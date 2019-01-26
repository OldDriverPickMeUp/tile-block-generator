import random

from dev.helpers import enable_debug_log
from dev.try_image_reader import get_example_reader
from dev.try_read_rules import get_matrix_data
from generator.core import Model2D, MatrixT2D
from generator.common import CanNotGenerateError


def find_a_seed_can_generate(size, file_prefix):
    mat_data, categories = get_matrix_data()
    mat_T = MatrixT2D(mat_data)
    reader = get_example_reader()
    for each in range(1000):
        random.seed()
        seed = random.randint(10000, 100000000)
        # seed = 48002114
        model = Model2D(size, categories, mat_T)
        try:
            loop = model.generate(seed, save_snapshot=False, reader=reader, save_prefix='out/example', max_loop=2000)
        except CanNotGenerateError as e:
            print('==', each, size, seed, 'fail', e)
            # break
            continue
        model.save_snapshot(reader, f'{file_prefix}-{size}-{seed}')
        print('==', 'find proper generated', loop, f'seed={seed}')
        break


if __name__ == '__main__':
    # enable_debug_log()
    find_a_seed_can_generate((20, 20), 'generate')
