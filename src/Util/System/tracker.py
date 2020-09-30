

import time, datetime
import os
import psutil

def _print_current_memory_percent():

    _process = psutil.Process(os.getpid())
    print(f"Memory usege : {_process.memory_percent()*100:.2}%")


def _elapsed_since(start):

    delta_seconds = time.time() - start
    delta_hours, remain_seconds = divmod(delta_seconds, 3600)
    #delta_hours = int(delta_hours)
    delta_minutes, remain_seconds = divmod(remain_seconds, 60)
    #delta_minutes = int(delta_hours)
    return f"{delta_hours:02.0f}:{delta_minutes:02.0f}:{remain_seconds:02.6f}"
    #return time.strftime( "%H:%M:%S", time.gmtime(delta_seconds) )


def _get_process_memory():
    r"""
    process memory in unit of [byte]
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def _track(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        mem_before = _get_process_memory()
        result = func(*args, **kwargs)
        mem_after = _get_process_memory()
        elapsed_time = _elapsed_since(start)
        sep = '\n  '
        print(f"{func.__name__}:{sep}memory before: {mem_before:,} bytes{sep}after        : {mem_after:,} bytes{sep}consumed     : {mem_after - mem_before:,} bytes{sep}exec time    : {elapsed_time}")
        return result
    return wrapper

@_track
def test_func(n, k):

    time.sleep(k)

    return [1.0,] * n

if __name__ == "__main__":
    _ = test_func(n=1, k=2)
