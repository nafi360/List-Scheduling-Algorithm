from __future__ import annotations
import os
import time
import math
import queue
import signal
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------- util beban CPU ----------

def busy_cpu(seconds: float) -> None:
    """Beban CPU murni selama 'seconds' detik (tanpa sleep)."""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < end:
        # kalkulasi dummy agar tetap di CPU
        x = math.fmod(x * 1.000001 + 3.14159, 1.0)

# ---------- worker ----------

@dataclass
class TaskMsg:
    task_id: int
    release_time: float  # absolute perf_counter saat task boleh mulai (setelah comm)
    duration: float      # durasi eksekusi CPU (detik)

@dataclass
class ResultMsg:
    task_id: int
    proc_id: int
    start: float
    finish: float


def _pin_to_core(core_id: int):
    try:
        os.sched_setaffinity(0, {int(core_id)})
    except Exception:
        pass  # kalau nggak bisa, lanjut tanpa pinning


def worker_loop(proc_id: int, core_id: int, in_q: mp.Queue, out_q: mp.Queue, stop_ev: mp.Event): # type: ignore
    _pin_to_core(core_id)
    # local availability
    available_time = time.perf_counter()
    while not stop_ev.is_set():
        try:
            msg: TaskMsg | None = in_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if msg is None:
            break
        # Tunggu sampai release_time (komunikasi selesai) dan worker available
        start_at = max(msg.release_time, available_time)
        now = time.perf_counter()
        if start_at > now:
            time.sleep(start_at - now)
        real_start = time.perf_counter()
        busy_cpu(msg.duration)
        real_finish = time.perf_counter()
        available_time = real_finish
        out_q.put(ResultMsg(task_id=msg.task_id, proc_id=proc_id, start=real_start, finish=real_finish))

# ---------- controller ----------

def execute_schedule(G, assign, exec_time, comm, cpu_cores: List[int] | None = None, processes: int = 8):
    """
    Jalankan DAG nyata sesuai mapping 'assign'.
    - exec_time[v, p] adalah durasi CPU target (detik) di prosesor p.
    - comm[u, v] adalah delay komunikasi (detik) ketika pindah prosesor.
    """
    import networkx as nx

    P = processes
    if cpu_cores is None:
        cpu_cores = list(range(P))
    assert len(cpu_cores) >= P

    manager = mp.Manager()
    in_queues = [manager.Queue() for _ in range(P)]
    out_q = manager.Queue()
    stop_ev = mp.Event()

    procs = []
    for p in range(P):
        w = mp.Process(target=worker_loop, args=(p, cpu_cores[p], in_queues[p], out_q, stop_ev), daemon=True)
        w.start()
        procs.append(w)

    # tracking
    indeg = {v: G.in_degree(v) for v in G.nodes}
    parents = {v: list(G.predecessors(v)) for v in G.nodes}
    finished: Dict[int, Tuple[float, float]] = {}  # v -> (start, finish)

    # antrian ready jika semua parent sudah finish
    import heapq
    ready_heap: List[Tuple[float, int]] = []  # (earliest_release_time, v)

    t0 = time.perf_counter()

    # inisialisasi node sumber
    for v, d in indeg.items():
        if d == 0:
            heapq.heappush(ready_heap, (t0, v))

    total = G.number_of_nodes()
    dispatched = set()

    while len(finished) < total:
        # ambil hasil jika ada
        try:
            while True:
                res: ResultMsg = out_q.get_nowait()
                finished[res.task_id] = (res.start, res.finish)
        except queue.Empty:
            pass

        # coba dispatch yang ready
        while ready_heap and len(finished) < total:
            release_time, v = heapq.heappop(ready_heap)
            if v in dispatched:
                continue
            p = int(assign[v])
            dur = float(exec_time[v, p])
            in_queues[p].put(TaskMsg(task_id=v, release_time=release_time, duration=dur))
            dispatched.add(v)

        # cek node yang jadi ready karena semua parent sudah selesai
        for v in G.nodes:
            if v in dispatched:
                continue
            if all(pr in finished for pr in parents[v]):
                # hitung release time = max(parent_finish + comm_delay)
                rel = t0
                for u in parents[v]:
                    ufinish = finished[u][1]
                    rel = max(rel, ufinish + (0.0 if assign[u] == assign[v] else float(comm[u, v])))
                heapq.heappush(ready_heap, (rel, v))

        time.sleep(0.01)

    # matikan workers
    stop_ev.set()
    for q in in_queues:
        q.put(None)
    for w in procs:
        w.join()

    # hasil jadwal nyata
    import numpy as np
    n = exec_time.shape[0]
    start = np.zeros(n)
    finish = np.zeros(n)
    for v, (s, f) in finished.items():
        start[v] = s - t0
        finish[v] = f - t0
    return start, finish