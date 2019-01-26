import os

from generator.readers import ImageReader

filepath = os.path.abspath(__file__)

example_tile_path = os.path.join(os.path.dirname(os.path.dirname(filepath)), 'examples/tiles')
tile_filenames = [os.path.join(example_tile_path, each) for each in os.listdir(example_tile_path) if
                  each.endswith('.png')]


def get_example_reader():
    return ImageReader(tile_filenames)


if __name__ == '__main__':
    get_example_reader().preview()
