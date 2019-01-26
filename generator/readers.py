import abc

import numpy as np


class _MatrixTReader(metaclass=abc.ABCMeta):
    def __init__(self):
        self._mat = None
        self.w = None
        self.h = None
        self._d = None

    @abc.abstractmethod
    def load_example(self, mat):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_rules(self):
        raise NotImplementedError


class MatrixTReader2D(_MatrixTReader):
    def load_example(self, mat):
        self._mat = np.matrix(mat)
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
        for x, y in self._all_matrix_points():
            current_category = self._mat[x, y]
            x_p = (x + 1, y)
            x_r_p = (x - 1, y)
            y_p = (x, y + 1)
            y_r_p = (x, y - 1)
            all_points = [x_p, x_r_p, y_p, y_r_p]
            for rules, point in zip(all_rules, all_points):
                if self._in_scope(point):
                    rules.append(self._get_rule(point, current_category))

        return all_rules
