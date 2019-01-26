from enum import Enum, auto

from generator.common import CanNotGenerateError


class ActionStatus(Enum):
    INITIALIZED = auto()
    EXECUTED = auto()
    ROLLBACK = auto()


class Action:
    def exec(self, model):
        raise NotImplementedError

    def rollback(self, model):
        raise NotImplementedError

    def create_new_actions(self, model):
        raise NotImplementedError


class PointSelect(Action):
    def __init__(self, point):
        self.point = point

    def exec(self, model):
        model.exec_point_select(self.point)
        return True

    def rollback(self, model):
        model.rollback_to_last_point()

    def create_new_actions(self, model):
        return model.create_new_category_select_actions()


class CategorySelect(Action):
    def __init__(self, category):
        self.category = category

    def exec(self, model):
        return model.exec_category_select(self.category)

    def rollback(self, model):
        model.rollback_category_select()

    def create_new_actions(self, model):
        return model.create_new_point_select_actions()


class Node:
    def __init__(self, parent):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def rollback_action(self, model):
        raise NotImplementedError


class RootNode(Node):
    def __init__(self):
        super().__init__(parent=None)

    def get_can_exec_nodes(self):
        return [each for each in self.children if each.status == ActionStatus.INITIALIZED]

    def rollback_action(self,model):
        raise CanNotGenerateError()


class ActionNode(Node):
    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.status = ActionStatus.INITIALIZED
        self.action = action

    def exec_action(self, model):
        self.status = ActionStatus.EXECUTED
        return self.action.exec(model)

    def rollback_action(self, model):
        self.status = ActionStatus.ROLLBACK
        return self.action.rollback(model)

    def get_can_exec_nodes(self):
        return [each for each in self.children if each.status == ActionStatus.INITIALIZED]

    def create_new_action_nodes(self, model):
        actions = self.action.create_new_actions(model)
        for each in actions:
            self.add_child(ActionNode(each, self))
