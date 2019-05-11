import logging
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dev.helpers import enable_debug_log, enable_info_log
from dev.try_image_reader import get_example_reader
from dev.try_read_rules import get_matrix_data
from generator.core import MatrixT2D, Overlap2D
from generator.common import CanNotGenerateError


def find_a_seed_can_generate(output_size, overlap_units, block_size, file_prefix):
    mat_data, categories = get_matrix_data()
    mat_T = MatrixT2D(mat_data)
    reader = get_example_reader()
    for each in range(1000):
        random.seed()
        seed = random.randint(10000, 100000000)
        # print(seed)
        # seed = 4252051
        model = Overlap2D(output_size, overlap_units, block_size, categories, mat_T)
        try:
            model.generate(seed, save_snapshot=False, reader=reader, save_prefix='out/example', max_loop=300,
                           max_block_loop=100, save_block_snapshot=False)
        except CanNotGenerateError:
            print('==', each, output_size, block_size, overlap_units, seed, 'fail')
            continue
        except Exception as e:
            # model.save_snapshot(reader, f'{file_prefix}-{output_size}-{block_size}-{overlap_units}-{seed}')
            raise e
        model.save_snapshot(reader, f'{file_prefix}-{output_size}-{block_size}-{overlap_units}-{seed}')
        print('==', 'find proper generated')
        break


if __name__ == '__main__':
    # enable_debug_log()
    enable_info_log()
    find_a_seed_can_generate((40, 40), 7, (8, 8), 'overlap')
