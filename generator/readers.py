import abc

import numpy as np
from PIL import Image, ImageColor, ImageDraw


class _MatrixTReader(metaclass=abc.ABCMeta):
    def __init__(self):
        self._mat = None
        self.w = None
        self.h = None
        self.d = None

    @abc.abstractmethod
    def load_example(self, mat):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_rules(self):
        raise NotImplementedError


class MatrixTReader2D(_MatrixTReader):
    def load_example(self, mat):
        self._mat = np.matrix(mat).T
        if len(self._mat.shape) != 2:
            raise Exception('need 2D example')
        self.w, self.h = self._mat.shape

    def _in_scope(self, point):
        x, y = point
        return 0 <= x < self.w and 0 <= y < self.h

    def _all_matrix_points(self):
        for x in range(self.w):
            for y in range(self.h):
                yield x, y

    def _get_rule(self, target_point, current_category):
        x, y = target_point
        another_category = self._mat[x, y]
        return current_category, another_category, True

    def generate_rules(self):
        x_rules = []
        x_reverse_rules = []
        y_rules = []
        y_reverse_rules = []
        all_rules = [x_rules, x_reverse_rules, y_rules, y_reverse_rules]
        categories_set = set()
        for x, y in self._all_matrix_points():
            current_category = self._mat[x, y]
            categories_set.add(current_category)
            x_p = (x + 1, y)
            x_r_p = (x - 1, y)
            y_p = (x, y + 1)
            y_r_p = (x, y - 1)
            all_points = [x_p, x_r_p, y_p, y_r_p]
            for rules, point in zip(all_rules, all_points):
                if self._in_scope(point):
                    rules.append(self._get_rule(point, current_category))

        return all_rules, categories_set


class ImageReader:
    def __init__(self, filenames, allow_empty=True):
        self.allow_empty = allow_empty
        self.filenames = filenames
        self.images = [Image.open(fn) for fn in self.filenames]
        self.size = self._valid_size()
        if self.allow_empty:
            empty = Image.new(mode='RGBA', size=self.size, color=ImageColor.getrgb("#ffffff"))
            self.images.insert(0, empty)

    def _valid_size(self):
        size = None
        try:
            for each in self.images:
                if size is None:
                    size = each.size
                    continue
                assert size == each.size
        except AssertionError:
            for each in self.images:
                each.close()
            raise Exception("different size for each tile")
        return size

    def preview(self):
        count = len(self.images)
        columns = 10
        lines = count // columns
        last = count % columns
        if last != 0:
            lines += 1
        w, h = self.size
        width = w * columns
        line_height = h + 20
        height = lines * line_height
        new_image = Image.new(mode='RGBA', size=(width, height),
                              color=ImageColor.getcolor('rgba(1,1,1,1)', mode='RGBA'))
        cursor = (0, 0)
        new_draw = ImageDraw.Draw(new_image)
        category_id = 0
        text_color = ImageColor.getcolor('rgb(0,0,0)', mode='RGB')
        for im in self.images:
            new_image.paste(im, cursor)
            new_draw.text((cursor[0] + w // 2, cursor[1] + h), f'{category_id}', text_color)
            category_id += 1
            cursor = (cursor[0] + w, cursor[1])
            x, y = cursor
            if x >= width:
                x = 0
                y += line_height
                cursor = (x, y)
        new_image.show()

    def __getitem__(self, item):
        if item < 0:
            raise IndexError
        im = self.images[item]
        return im.copy()

    def __len__(self):
        return len(self.images)
