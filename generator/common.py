from enum import unique, Enum, auto


@unique
class Direction(Enum):
    x = auto()
    x_reverse = auto()
    y = auto()
    y_reverse = auto()
    z = auto()
    z_reverse = auto()


class CanNotGenerateError(Exception):
    pass


class NoMorePoints(Exception):
    pass


class InitCatalogError(Exception):
    pass
