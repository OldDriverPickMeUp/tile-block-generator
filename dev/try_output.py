import numpy as np

from generator.output import Output
from dev.try_image_reader import get_example_reader


def get_example_mat1():
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 19, 19, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 9, 9, 3, 0, 0, 0, 0],
        [0, 1, 19, 15, 6, 6, 16, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]


def get_example_mat2():
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 9, 3, 0, 0, 0, 0],
        [0, 0, 0, 8, 15, 6, 16, 3, 0, 0, 0],
        [0, 0, 8, 15, 6, 6, 6, 16, 3, 0, 0],
        [0, 1, 15, 4, 4, 4, 4, 4, 16, 19, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]


def concat(a, b):
    height = len(a)
    new_mat = [0] * height
    for i in range(height):
        new_row = []
        new_row.extend(a[i])
        new_row.extend(b[i])
        new_mat[i] = new_row
    return new_mat


def get_example_mat():
    return concat(get_example_mat1(), get_example_mat2())


# print(np.array(mat))
if __name__ == '__main__':
    output = Output(get_example_reader())
    output.show_mat(get_example_mat())
