import logging
import sys


def enable_debug_log():
    f = logging.Filter('tile-block-generator')
    h = logging.StreamHandler(sys.stdout)
    h.addFilter(f)
    logging.basicConfig(level=logging.DEBUG, handlers=[h], format='%(message)s')


def enable_info_log():
    f = logging.Filter('tile-block-generator')
    h = logging.StreamHandler(sys.stdout)
    h.addFilter(f)
    logging.basicConfig(level=logging.INFO, handlers=[h], format='%(message)s')
