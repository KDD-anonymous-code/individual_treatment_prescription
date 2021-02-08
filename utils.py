from queue import Full
import multiprocessing
import threading
from contextlib import contextmanager
from tqdm import tqdm


def split_treatment_control(df, treatment):
    return df[df[treatment] == 1], df[df[treatment] == 0]


class ProgressBar:
    _pbar = None  # mono-process
    _thread = None  # multi-process, main process
    _queue = None  # multi-process, main process and workers processes

    @staticmethod
    def init_monop(total, desc=None):
        ProgressBar._pbar = tqdm(total=total, desc=desc)

    @staticmethod
    @contextmanager
    def init_pool(n_cpus, total, desc=None):
        queue = multiprocessing.Queue()

        def func():
            pbar = tqdm(total=total, desc=desc)
            while True:
                i = queue.get()
                if i is None:
                    return
                pbar.update(i)
        ProgressBar._thread = threading.Thread(target=func, daemon=True)
        ProgressBar._thread.start()
        ProgressBar._queue = queue

        try:
            with multiprocessing.Pool(n_cpus, ProgressBar._init_multip_worker, initargs=(queue,)) as pool:
                yield pool
        finally:
            queue.put_nowait(None)
            ProgressBar._thread.join(timeout=2)

    @staticmethod
    def _init_multip_worker(queue):
        ProgressBar._queue = queue

    @staticmethod
    def incr(value=1):
        queue = ProgressBar._queue
        if queue is None:
            ProgressBar._pbar.update(value)
        else:
            try:
                queue.put_nowait(value)
            except Full:
                pass
