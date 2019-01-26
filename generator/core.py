import abc
import logging
import random

import pandas as pd
import numpy as np

from generator.common import CanNotGenerateError, NoMorePoints, InitCatalogError
from .actions import RootNode, PointSelect, ActionNode, CategorySelect
from .output import Output
from .common import Direction

logger = logging.getLogger('tile-block-generator')


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
        df = pd.DataFrame(records).drop_duplicates().set_index([0, 1])
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
        self._stack = []

    def _build_storage(self):
        return [[self.init_categories.copy() for _ in range(self.h)] for _ in range(self.w)]

    def update_point(self, point, can_tile_block_categories):
        x, y = point
        try:
            old_set = self._storage[x][y]
        except Exception as e:
            logging.debug(f'error {(x,y)}')
            raise e
        self._ahead_log.append((point, old_set))
        new_set = old_set.intersection(set(can_tile_block_categories))
        self._storage[x][y] = new_set

    def push_state(self):
        return self._stack.append(len(self._ahead_log))

    def pop_state(self):
        if len(self._stack) == 0:
            raise CanNotGenerateError('not more catalog to pop')
        state = self._stack.pop()
        to_restore_log = self._ahead_log[state:]
        remain_log = self._ahead_log[:state]
        self._ahead_log = remain_log
        for point, allow_set in reversed(to_restore_log):
            x, y = point
            self._storage[x][y] = allow_set

    def get_can_select_set(self, point):
        x, y = point
        return list(self._storage[x][y])


class _Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self, seed):
        raise NotImplementedError


class Model2D(_Model):
    def __init__(self, size, categories, matrix_T, on_filter=None):
        self.w, self.h = size
        self.categories = categories
        self._on_filter_cb = on_filter
        self.matrix_T = matrix_T

        self._catalog = None
        self._storage = None
        self._stack = None
        self._current_node = None
        self._action_tree = None
        self._current_point = None

    def _build_for_generate(self, init_storage, init_action_tree=None):
        self._catalog = Catalog2D((self.w, self.h), self.categories)
        if init_storage is not None:
            self._storage = np.array(init_storage, dtype=np.int)
            assert self._storage.shape == (self.w, self.h)
            existed = np.argwhere(self._storage != -1)
            for each in existed:
                x, y = each
                category = self._storage[x, y]
                for p, direction in self._get_neighbor_with_directions(each):
                    will_merge_categories = self._filter_category(p,
                                                                  self.matrix_T.can_tile_categories(direction,
                                                                                                    category))
                    if len(will_merge_categories) == 0:
                        raise InitCatalogError()
                    self._catalog.update_point(p, will_merge_categories)
        else:
            s = np.ndarray(shape=(self.w, self.h), dtype=np.int)
            s[:] = -1
            self._storage = s
        if init_action_tree is not None:
            self._action_tree = init_action_tree
            self._current_node = self._action_tree
        else:
            self._action_tree = RootNode()
            self._current_node = self._action_tree
            for action in self.create_new_point_select_actions():
                self._action_tree.add_child(ActionNode(action, self._current_node))
            if len(self._current_node.get_can_exec_nodes()) == 0:
                raise NoMorePoints()

    def in_scope(self, point):
        x, y = point
        return 0 <= x < self.w and 0 <= y < self.h

    def _get_neighbor_with_directions(self, point):
        x, y = point
        return [((_x, _y), _d) for _x, _y, _d in
                [(x - 1, y, Direction.x_reverse), (x + 1, y, Direction.x), (x, y - 1, Direction.y_reverse),
                 (x, y + 1, Direction.y)]
                if self.in_scope((_x, _y))]

    def _get_neighbors(self, point):
        x, y = point
        return [(_x, _y) for _x, _y, in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] if self.in_scope((_x, _y))]

    def generate(self, seed=None, save_snapshot=False, reader=None, save_prefix=None, init_storage=None,
                 init_action_tree=None, max_loop=20000):
        if seed is not None:
            random.seed(seed)
        self._build_for_generate(init_storage, init_action_tree)
        loop = 0
        while True:
            logger.debug(f'== {loop}')
            if loop > max_loop:
                raise CanNotGenerateError(f'reach max loop {max_loop}')
            try:
                self._generate()
                if save_snapshot:
                    self.save_snapshot(reader, f'{save_prefix}-{loop}')
            except NoMorePoints:
                break
            loop += 1
        return loop

    def _generate(self):
        can_exec_nodes = self._current_node.get_can_exec_nodes()
        if len(can_exec_nodes) == 0:
            self._current_node.rollback_action(self)
            self._current_node = self._current_node.parent
            logging.debug(f'== rollback at {self._current_point}')
            return
        selected_node = random.choice(can_exec_nodes)
        self._current_node = selected_node
        status = self._current_node.exec_action(self)
        if status:
            self._current_node.create_new_action_nodes(self)
            return
        self._current_node.rollback_action(self)
        self._current_node = self._current_node.parent

    def exec_point_select(self, point):
        self._current_point = point

    def rollback_to_last_point(self):
        if isinstance(self._current_node.action, PointSelect):
            parent = self._current_node.parent
            if parent is self._action_tree:
                self._current_point = None
                return
            parent = parent.parent
            self._current_point = parent.action.point
            return
            # rollback to root

        if isinstance(self._current_node.action, CategorySelect):
            parent = self._current_node.parent
            self._current_point = parent.action.point
            return
        self._current_point = None

    def create_new_category_select_actions(self):
        can_select_categories = self._catalog.get_can_select_set(self._current_point)
        return [CategorySelect(each) for each in can_select_categories]

    def exec_category_select(self, category):
        x, y = self._current_point
        actions = []
        for p, direction in self._get_neighbor_with_directions(self._current_point):
            will_merge_categories = self._filter_category(p,
                                                          self.matrix_T.can_tile_categories(direction, category))
            current_categories = set(self._catalog.get_can_select_set(p))
            if len(current_categories.intersection(set(will_merge_categories))) == 0:
                logger.debug(f'{p} {direction}')
                return False
            actions.append((p, will_merge_categories))

        self._storage[x, y] = category
        self._catalog.push_state()
        for p, categories in actions:
            self._catalog.update_point(p, categories)
        return True

    def rollback_category_select(self):
        x, y = self._current_point
        if self._storage[x, y] == -1:
            return
        self._catalog.pop_state()
        self._storage[x, y] = -1

    # def create_new_point_select_actions(self):
    #     points = [each for each in self._get_neighbors(self._current_point) if self._storage[each[0], each[1]] == -1]
    #     if len(points) == 0:
    #         points = np.argwhere(self._storage == -1)
    #     if len(points) == 0:
    #         raise NoMorePoints()
    #     return [PointSelect(each) for each in points if self._storage[each[0], each[1]] == -1]

    def create_new_point_select_actions(self):
        handled_points = np.argwhere(self._storage != -1)
        if len(handled_points) == 0:
            return [PointSelect(p) for p in np.argwhere(self._storage == -1)]
        record = np.zeros(self._storage.shape, dtype=np.bool)
        for each in handled_points:
            record[each[0], each[1]] = True
        all_neighbours = []
        for p in handled_points:
            neighbours = self._get_neighbors(p)
            for each in neighbours:
                if record[each[0], each[1]]:
                    continue
                record[each[0], each[1]] = True
                all_neighbours.append(each)
        if len(all_neighbours) == 0:
            raise NoMorePoints()
        cnts_mt1 = []
        cnts_eq1 = []
        for each in all_neighbours:
            neighbour_count = len([p for p in self._get_neighbors(each) if self._storage[p[0], p[1]] != -1])
            if neighbour_count > 1:
                cnts_mt1.append(each)
                continue
            cnts_eq1.append(each)
        if len(cnts_mt1) > 0:
            return [PointSelect(p) for p in cnts_mt1]
        return [PointSelect(p) for p in cnts_eq1]

    def get_mat(self):
        return np.matrix(self._storage)

    @property
    def storage(self):
        return self._storage

    @property
    def action_tree(self):
        return self._action_tree

    def save_snapshot(self, reader, name):
        out_mat = self.get_mat().T
        Output(reader).save_mat(out_mat, name)

    def _filter_category(self, point, can_select_categories):
        if callable(self._on_filter_cb):
            return self._on_filter_cb(self, point, can_select_categories)
        return can_select_categories

    @property
    def size(self):
        return self.w, self.h


class StorageStack:
    def __init__(self, storage):
        self._stack = []
        self._storage = storage

    def push(self, cursor_point, block_size, model):
        self._stack.append((cursor_point, block_size, model))

    def pop(self):
        return self._stack.pop()

    @property
    def storage(self):
        s = self._storage.copy()
        for cursor_point, block_size, model in self._stack:
            cx, cy = cursor_point
            bw, bh = block_size
            s[cx:cx + bw, cy:cy + bh] = model.storage
        return s

    @property
    def current(self):
        return self._stack[-1]


class Overlap2D(_Model):
    def __init__(self, output_size, overlap_units, block_size, categories, matrix_T, on_filter=None):
        self.categories = categories
        self.block_size = block_size
        self.matrix_T = matrix_T
        self.overlap_units = overlap_units
        self.on_filter_cb = on_filter
        self.w, self.h = output_size

        self._cursor_point = None
        self._current_block_size = None
        self._storage_stack = None

    def _build_for_generate_block(self):
        cx, cy = self._cursor_point
        bw, bh = self.block_size
        rw = self.w - cx if cx + bw > self.w else bw
        rh = self.h - cy if cy + bh > self.h else bh
        self._current_block_size = (rw, rh)

    def _generate_block(self, save_snapshot=False, reader=None, save_prefix=None, max_loop=20000, action_tree=None):
        self._build_for_generate_block()
        rw, rh = self._current_block_size
        cx, cy = self._cursor_point
        init_storage = self._storage_stack.storage[cx:cx + rw, cy:cy + rh]
        model = Model2D(self._current_block_size, self.categories, self.matrix_T, self.on_filter_cb)
        try:
            loop = model.generate(None, save_snapshot, reader, save_prefix, init_storage, action_tree, max_loop)
            logger.info(f'== block {self._cursor_point},{self._current_block_size} generated with loop={loop}')
        except CanNotGenerateError as e:
            logger.error(f'== block {self._cursor_point},{self._current_block_size} failed {e}')
            return False
        except InitCatalogError:
            return False
        self._storage_stack.push(self._cursor_point, self._current_block_size, model)
        return True

    def _build_for_generate(self, init_storage):
        self._cursor_point = (0, 0)
        if init_storage is None:
            storage = np.ndarray(shape=(self.w, self.h), dtype=np.int)
            storage[:] = -1
            self._storage_stack = StorageStack(storage)
            return
        storage = np.array(init_storage, dtype=np.int)
        assert storage.shape == (self.w, self.h)
        self._storage_stack = StorageStack(storage)

    def generate(self, seed=None, save_snapshot=False, reader=None, save_prefix=None, init_storage=None,
                 max_loop=20000, max_block_loop=30000, save_block_snapshot=False):
        if seed is not None:
            random.seed(seed)
        bw, bh = self.block_size
        move_x_step = bw - self.overlap_units
        move_y_step = bh - self.overlap_units
        assert move_x_step > 0 and move_y_step > 0
        self._build_for_generate(init_storage)
        block = 1
        last_action_tree = None
        loop = 0
        while True:
            logger.info(f'== block loop {loop}')
            if loop > max_block_loop:
                raise CanNotGenerateError(f'reach max block loop {max_block_loop}')
            status = self._generate_block(save_snapshot, reader, f'{save_prefix}-{block}', max_loop, last_action_tree)
            if save_block_snapshot:
                self.save_snapshot(reader, f'{save_prefix}-{block}')

            loop += 1
            if not status:
                if block == 1:
                    raise CanNotGenerateError()
                while True:
                    self._cursor_point, self._current_block_size, last_model = self._storage_stack.pop()
                    last_action_tree = last_model.action_tree
                    if len(last_action_tree.get_can_exec_nodes()) != 0:
                        break
                continue
            last_action_tree = None
            cx, cy = self._cursor_point
            if cx + bw >= self.w:
                if cy + bh >= self.h:
                    break
                cx = 0
                cy += move_y_step
            else:
                cx += move_x_step

            self._cursor_point = (cx, cy)
            block += 1

    def get_mat(self):
        return np.matrix(self._storage_stack.storage)

    def save_snapshot(self, reader, name):
        out_mat = self.get_mat().T
        Output(reader).save_mat(out_mat, name)
