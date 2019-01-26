import abc
import random

import pandas as pd

from .common import Direction


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
        self._ahead_log = []

    def _build_storage(self):
        return [[self.init_categories.copy() for _ in range(self.h)] for _ in range(self.w)]

    def update_point(self, point, can_tile_block_categories):
        x, y = point
        old_set = self._storage[x][y]
        self._ahead_log.append((point, old_set))
        new_set = old_set.intersection(set(can_tile_block_categories))
        self._storage[x][y] = new_set

    def forbidden_at_point(self, point, forbidden_category):
        x, y = point
        old_set = self._storage[x][y]
        if forbidden_category not in old_set:
            raise Exception(f'should have category {forbidden_category} in old set')
        self._storage[x][y].discard(forbidden_category)

    def get_state(self):
        return len(self._ahead_log)

    def restore_state(self, state):
        to_restore_log = self._ahead_log[state:]
        remain_log = self._ahead_log[:state]
        self._ahead_log = remain_log
        for point, allow_set in to_restore_log.reverse():
            x, y = point
            self._storage[x][y] = allow_set

    def get_can_select_set(self, point):
        x, y = point
        return self._storage[x][y]


class PointQueue:
    def __init__(self):
        self._data = []
        self._processed_cursor = 0
        self._add_cursor = 1

    def append(self, data):
        self._data.append(data)
        self._add_cursor += 1

    def extend(self, data_list):
        self._data.extend(data_list)
        self._add_cursor += len(data_list)

    def get_state(self):
        return self._processed_cursor, self._add_cursor

    def restore_state(self, state):
        self._processed_cursor, self._add_cursor = state
        to_rollback = self._data[self._add_cursor:]
        self._data = self._data[:self._add_cursor]
        return to_rollback

    def pop_one(self):
        cursor = self._processed_cursor
        self._processed_cursor += 1
        return self._data[cursor]

    def get_current(self):
        return self._data[self._processed_cursor]


class _Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self, seed):
        raise NotImplementedError


class Model2D(_Model):
    def __init__(self, size, categories, matrix_T, on_select=None, on_filter=None):
        self.w, self.h = size
        self.categories = categories
        self._on_select_cb = on_select
        self._on_filter_cb = on_filter
        self._matrix_T = matrix_T

        self._catalog = None
        self._storage = None
        self._processed = None
        self._stack = None
        self._queue = None

    def _build_for_each_generate(self):
        self._catalog = Catalog2D((self.w, self.h), self.categories)
        self._storage = [[-1 for _ in range(self.h)] for _ in range(self.w)]
        self._processed = [[False for _ in range(self.h)] for _ in range(self.w)]
        self._stack = []
        self._queue = PointQueue()

    def _in_scope(self, point):
        x, y = point
        return 0 <= x < self.w and 0 <= y < self.h

    def _has_been_processed(self, point):
        x, y = point
        return self._processed[x][y]

    def _get_start_point(self):
        return random.randint(0, self.w - 1), random.randint(0, self.h - 1)

    def _get_neighbor_with_directions(self, point):
        x, y = point
        return [((_x, _y), _d) for _x, _y, _d in
                [(x - 1, y, Direction.x_reverse), (x + 1, y, Direction.x), (x, y - 1, Direction.y_reverse),
                 (x, y + 1, Direction.y)]
                if self._in_scope((x, y)) and not self._has_been_processed((x, y))]

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
        self._processed[x][y] = True
        return True, neighbors

    def _push_state(self):
        queue_state = self._queue.get_state()
        catalog_state = self._catalog.get_state()
        self._stack.append((queue_state, catalog_state))

    def _pop_state(self):
        queue_state, catalog_state = self._stack.pop()
        rollback_points = self._queue.restore_state(queue_state)
        for x, y in rollback_points:
            self._processed[x][y] = False
        self._catalog.restore_state(catalog_state)
        x, y = self._queue.get_current()
        current_category = self._storage[x][y]
        self._catalog.forbidden_at_point((x, y), current_category)

    def generate(self, seed):
        random.seed(seed)
        self._build_for_each_generate()
        start_point = self._get_start_point()
        self._queue.append(start_point)
        while True:
            self._push_state()
            try:
                point = self._queue.pop_one()
            except IndexError:
                break
            status, neighbors = self._generate_block(point)
            if not status:
                self._pop_state()
                continue
            self._queue.extend(neighbors)

    def _select_category(self, can_select_categories):
        if callable(self._on_select_cb):
            return self._on_select_cb(self, can_select_categories)
        return random.choice(can_select_categories)

    def _filter_category(self, can_select_categories):
        if callable(self._on_filter_cb):
            return self._on_filter_cb(self, can_select_categories)
        return can_select_categories
