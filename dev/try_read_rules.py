import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.try_output import get_example_mat
from generator.core import MatrixData
from generator.readers import MatrixTReader2D


def get_matrix_data():
    reader = MatrixTReader2D()
    reader.load_example(get_example_mat())
    rules, categories = reader.generate_rules()
    return MatrixData(rules), list(categories)


if __name__ == '__main__':
    mat_data, categories = get_matrix_data()
    print('==', 'categories')
    print(categories)
    print('==', 'x')
    print(mat_data.x)
    print('==', 'x_r')
    print(mat_data.x_reverse)
    print('==', 'y')
    print(mat_data.y)
    print('==', 'y_r')
    print(mat_data.y_reverse)
