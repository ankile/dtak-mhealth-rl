from contextlib import AbstractContextManager
import multiprocessing
from typing import Any, Callable


class OptionalPool(AbstractContextManager):
    """
    Context manager Pool that allows for completely turning off
    parallelization by setting n_processes to 1.
    """

    def __init__(self, processes: int = 1):
        self.processes = processes
        self.pool = None

    def __enter__(self):
        if self.processes > 1:
            self.pool = multiprocessing.Pool(processes=self.processes)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()

    def imap(self, func: Callable[..., Any], iterable):
        if self.pool is not None:
            return self.pool.imap(func, iterable)
        else:
            return map(func, iterable)
