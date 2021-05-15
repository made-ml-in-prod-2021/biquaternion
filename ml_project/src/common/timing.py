#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


if __name__ == '__main__':
    pass
