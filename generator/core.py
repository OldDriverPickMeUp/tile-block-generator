import abc
import random

import pandas as pd
from enum import Enum, auto, unique


@unique
class Direction(Enum):
    x = auto()
    x_reverse = auto()
    y = auto()
    y_reverse = auto()
    z = auto()
    z_reverse = auto()


class MatrixData:
    def __init__(self, all_records):
        self.x = self._build_dataframe(all_records[0])
        self.x_reverse = self._build_dataframe(all_records[1])
        self.y = self._build_dataframe(all_records[2])
        self.y_reverse = self._build_dataframe(all_records[3])
        if len(all_records) > 4:
            self.z = self._build_dataframe(all_records[4])
            self.z_reverse = self._build_dataframe(all_records[5])

    @staticmethod
    def _build_dataframe(records):
        df = pd.DataFrame(records).set_index([0, 1])
        df = df[df.columns[0]].astype(bool)
        return df.unstack(fill_value=False)


class _MatrixT(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def can_tile_categories(self, direction, tile_block_category):
        raise NotImplementedError


class MatrixT2D(_MatrixT):
    def __init__(self, mat):
        self._mat = mat

    @staticmethod
    def _find_true_set(row):
        return set(row[row == True].index.tolist())

    def can_tile_categories(self, direction, tile_block_category):
        if direction not in Direction:
            raise Exception('unexpected direction')
        mat = getattr(self._mat, direction.name)
        try:
            row = mat.loc[tile_block_category]
            return self._find_true_set(row)
        except IndexError:
            pass

        try:
            row = mat[tile_block_category]
            return self._find_true_set(row)
        except KeyError:
            pass
        return []


class _CatalogMatrix(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update_point(self, point, can_tile_block_categories):
        raise NotImplementedError


class Catalog2D(_CatalogMatrix):
    def __init__(self, size, categories):
        self.w, self.h = size
        self.init_categories = set(categories)
        self._storage = self._build_storage()

    def _build_storage(self):
        return [[self.init_categories.copy() for _ in range(self.h)] for _ in range(self.w)]

    def update_point(self, point, can_tile_block_categories):
        x, y = point
        old_set = self._storage[x][y]
        new_set = old_set.intersection(set(can_tile_block_categories))
        self._storage[x][y] = new_set

    def get_can_select_set(self, point):
        x, y = point
        return self._storage[x][y]


class PointQueue:
    def __init__(self):
        self._data = []
        self._processed_cursor = 0
        self._add_cursor = 1
        self._stack = []

    def append(self, data):
        self._data.append(data)
        self._add_cursor += 1

    def extend(self, data_list):
        self._data.extend(data_list)
        self._add_cursor += len(data_list)

    def push_state(self):
        self._stack.append((self._processed_cursor, self._add_cursor))

    def pop_state(self):
        self._processed_cursor, self._add_cursor = self._stack.pop()
        self._data = self._data[:self._add_cursor]

    def get_one(self):
        cursor = self._processed_cursor
        self._processed_cursor += 1
        return self._data[cursor]


class _Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self, seed):
        raise NotImplementedError


class Model2D(_Model):
    def __init__(self, size, categories, matrix_T, on_select=None, on_filter=None):
        self.w, self.h = size
        self._catalog = Catalog2D(size, categories)
        self._matrix_T = matrix_T
        self._storage = self._build_storage()
        self._on_select_cb = on_select
        self._on_filter_cb = on_filter

    def _build_storage(self):
        return [[-1 for _ in range(self.h)] for _ in range(self.w)]

    def _in_scope(self, point):
        x, y = point
        return 0 <= x < self.w and 0 <= y < self.h

    def _get_start_point(self):
        return random.randint(0, self.w - 1), random.randint(0, self.h - 1)

    def _get_neighbor_with_directions(self, point):
        x, y = point
        return [((_x, _y), _d) for _x, _y, _d in
                [(x - 1, y, Direction.x_reverse), (x + 1, y, Direction.x), (x, y - 1, Direction.y_reverse),
                 (x, y + 1, Direction.y)]
                if self._in_scope((x, y))]

    def _generate_block(self, point):
        can_select_categories = self._catalog.get_can_select_set(point)
        can_select_categories = self._filter_category(can_select_categories)
        neighbors = []
        if len(can_select_categories) == 0:
            return False, neighbors
        category = self._select_category(can_select_categories)
        x, y = point
        self._storage[x][y] = category
        for p, direction in self._get_neighbor_with_directions(point):
            self._catalog.update_point(p,
                                       self._filter_category(self._matrix_T.can_tile_categories(direction, category)))
            neighbors.append(p)
        return True, neighbors

    def generate(self, seed):
        random.seed(seed)
        start_point = self._get_start_point()
        queue = PointQueue()
        queue.append(start_point)
        while True:
            queue.push_state()
            try:
                point = queue.get_one()
            except IndexError:
                break
            status, neighbors = self._generate_block(point)
            if not status:
                queue.pop_state()
                continue
            queue.extend(neighbors)

    def _select_category(self, can_select_categories):
        if callable(self._on_select_cb):
            return self._on_select_cb(self, can_select_categories)
        return random.choice(can_select_categories)

    def _filter_category(self, can_select_categories):
        if callable(self._on_filter_cb):
            return self._on_filter_cb(self, can_select_categories)
        return can_select_categories
