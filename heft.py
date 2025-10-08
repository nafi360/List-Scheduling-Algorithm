import numpy as np
import networkx as nx

def _upward_rank(G: nx.DiGraph, exec_time: np.ndarray, comm: np.ndarray) -> np.ndarray:
    n = exec_time.shape[0]
    avg_exec = exec_time.mean(axis=1)
    r = np.zeros(n)
    for v in reversed(list(nx.topological_sort(G))):
        succ = list(G.successors(v))
        if succ:
            r[v] = avg_exec[v] + max(comm[v, w] + r[w] for w in succ)
        else:
            r[v] = avg_exec[v]
    return r


def heft_schedule(G: nx.DiGraph, exec_time: np.ndarray, comm: np.ndarray):
    n, P = exec_time.shape
    r = _upward_rank(G, exec_time, comm)
    order = list(range(n))
    order.sort(key=lambda v: r[v], reverse=True)

    avail = np.zeros(P)
    start = np.zeros(n)
    finish = np.zeros(n)
    assign = -np.ones(n, dtype=int)

    for v in order:
        best_p, best_finish, best_start = None, float("inf"), 0.0
        for p in range(P):
            ready = float(avail[p])
            for u in G.predecessors(v):
                comm_delay = 0.0 if assign[u] == p else float(comm[u, v])
                ready = max(ready, float(finish[u]) + comm_delay)
            f = ready + float(exec_time[v, p])
            if f < best_finish:
                best_finish, best_p, best_start = f, p, ready
        assign[v] = int(best_p)
        start[v] = float(best_start)
        finish[v] = float(best_finish)
        avail[best_p] = float(best_finish)
    return assign, start, finish