# heft.py
import numpy as np

def upward_rank(G, exec_time, comm):
    n = exec_time.shape[0]
    avg_exec = exec_time.mean(axis=1)
    r = np.zeros(n)
    order = list(reversed(list(nx.topological_sort(G)))) # type: ignore
    for v in order:
        succ = list(G.successors(v))
        if succ:
            r[v] = avg_exec[v] + max(comm[v, w] + r[w] for w in succ)
        else:
            r[v] = avg_exec[v]
    return r

def heft_schedule(G, exec_time, comm):
    import networkx as nx
    n, P = exec_time.shape
    r = upward_rank(G, exec_time, comm)
    order = list(range(n))
    order.sort(key=lambda v: r[v], reverse=True)

    avail = np.zeros(P)
    start = np.zeros(n)
    finish = np.zeros(n)
    assign = -np.ones(n, dtype=int)

    for v in order:
        best_p, best_finish, best_start = None, float('inf'), 0
        for p in range(P):
            # ready time = max availability proc & parents finish + comm if cross-proc
            ready = avail[p]
            for u in G.predecessors(v):
                comm_delay = 0 if assign[u] == p else comm[u, v]
                ready = max(ready, finish[u] + comm_delay)
            f = ready + exec_time[v, p]
            if f < best_finish:
                best_finish, best_p, best_start = f, p, ready
        assign[v] = best_p
        start[v] = best_start
        finish[v] = best_finish
        avail[best_p] = best_finish
    return assign, start, finish
