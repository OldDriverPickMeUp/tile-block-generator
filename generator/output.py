import numpy as np
from PIL import Image, ImageDraw, ImageColor


class Output:
    def __init__(self, reader):
        self._reader = reader

    def _generate_image(self, matrix):
        mat = np.matrix(matrix).T
        size = mat.shape
        w, h = self._reader.size
        im_size = (size[0] * w, size[1] * h)
        new_image = Image.new(mode='RGBA', size=im_size)
        mat_x, mat_y = size
        draw = ImageDraw.Draw(new_image)
        text_color = ImageColor.getrgb('#000000')
        for x in range(mat_x):
            for y in range(mat_y):
                pixel_x = x * w
                pixel_y = y * h
                category_id = mat[x, y]
                if category_id != -1:
                    im = self._reader[category_id]
                    new_image.paste(im, (pixel_x, pixel_y))
                draw.text((pixel_x + w // 2, pixel_y + h // 2), f'{category_id}', fill=text_color)
        return new_image

    def show_mat(self, matrix):
        self._generate_image(matrix).show()

    def save_mat(self, matrix, name):
        im = self._generate_image(matrix)
        im.save(f'{name}.png')
