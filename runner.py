from __future__ import annotations
import time
import math
import multiprocessing as mp
from collections import deque
from typing import Tuple
import networkx as nx
import numpy as np

SECONDS_PER_COMM_UNIT = 0.001  # 1 unit comm = 1 ms

class ProcessorPool:
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.pool = mp.Pool(processes=pool_size)
    def submit(self, fn, *args, **kwargs):
        return self.pool.apply_async(fn, args=args, kwds=kwargs)
    def close(self):
        self.pool.close(); self.pool.join()


def _sleep_comm_delay(delay_s: float):
    if delay_s > 0:
        time.sleep(delay_s)


def run_dag_realtime(G: nx.DiGraph,
                     assign: np.ndarray,
                     exec_time_model: np.ndarray,
                     comm: np.ndarray,
                     P: int) -> Tuple[np.ndarray, np.ndarray]:
    n = G.number_of_nodes()
    start = np.full(n, np.nan)
    finish = np.full(n, np.nan)

    indeg = {v: G.in_degree(v) for v in G.nodes}
    ready = deque([v for v, d in indeg.items() if d == 0])

    pool_size = min(mp.cpu_count(), max(1, P))
    pool = ProcessorPool(pool_size)
    running = {}

    try:
        while ready or running:
            while ready:
                v = ready.popleft()
                # hitung delay komunikasi maksimum dari parent ke v
                comm_delay = 0.0
                for u in G.predecessors(v):
                    if not math.isfinite(finish[u]):
                        comm_delay = 0.0
                        break
                    if int(assign[u]) != int(assign[v]):
                        comm_delay = max(comm_delay, comm[u, v] * SECONDS_PER_COMM_UNIT)
                _sleep_comm_delay(comm_delay)

                start[v] = time.perf_counter()
                # Jalankan task sesuai worker_id = assign % pool_size
                worker_id = int(assign[v]) % pool_size
                fut = pool.submit(_run_task_wrapper, v)
                running[v] = (worker_id, fut)

            done_list = []
            for v, (wid, fut) in list(running.items()):
                if fut.ready():
                    fut.get()
                    finish[v] = time.perf_counter()
                    done_list.append(v)
            for v in done_list:
                running.pop(v, None)
                for w in G.successors(v):
                    indeg[w] -= 1
                    if indeg[w] == 0:
                        ready.append(w)
            if not done_list:
                time.sleep(0.001)
    finally:
        pool.close()

    return start, finish

# dipisah supaya apply_async bisa picklable
from workload import run_task as _run_task_wrapper