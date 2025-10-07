# dag_gen.py
import numpy as np
import networkx as nx
rng = np.random.default_rng()

def gen_dag(n_nodes, shape, max_outdeg, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Levelize pakai shape: ekspektasi lebar ~ shape * sqrt(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    # Buat level
    n_levels = max(2, int(np.ceil(np.sqrt(n_nodes) / max(0.5, min(shape, 4)) )))
    levels = [[] for _ in range(n_levels)]
    # Distribusi node ke level
    for v in range(n_nodes):
        lv = min(n_levels-1, int(rng.normal(loc=n_levels/2, scale=n_levels/(2*shape))))
        lv = max(0, min(n_levels-1, lv))
        levels[lv].append(v)
    # Pastikan source di level 0, sink di level terakhir ada isi
    if not levels[0]: levels[0].append(0)
    if not levels[-1]: levels[-1].append(n_nodes-1)

    # Tambahkan edge dari level i ke j>i, kontrol out-degree
    for i in range(n_levels-1):
        for u in levels[i]:
            outdeg = rng.integers(0, max_outdeg+1)
            targets = []
            # pilih beberapa level di depan
            for _ in range(outdeg):
                j = rng.integers(i+1, n_levels)
                if levels[j]:
                    v = int(rng.choice(levels[j]))
                    if u != v and not G.has_edge(u, v):
                        targets.append(v)
            for v in set(targets):
                G.add_edge(u, v)
    # Remove cycles (preventive), ensure DAG
    assert nx.is_directed_acyclic_graph(G)
    # Pastikan konektivitas dari source ke sink: tambahkan edge jika perlu
    # (simple fix: connect isolated levels)
    for i in range(n_levels-1):
        if not any((u,v) in G.edges for u in levels[i] for v in levels[i+1]):
            u = rng.choice(levels[i])
            v = rng.choice(levels[i+1])
            G.add_edge(int(u), int(v))
    return G, levels

def assign_costs(G, P, ccr, comp_cost_range=(0.5, 1.5), seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = G.number_of_nodes()
    base = rng.uniform(1.0, 10.0, size=n)  # base task cost
    # heterogenitas prosesor
    proc_factor = rng.uniform(0.7, 1.3, size=P)
    exec_time = np.zeros((n, P))
    for v in range(n):
        lo, hi = comp_cost_range
        jitter = rng.uniform(lo, hi)
        for p in range(P):
            exec_time[v, p] = base[v] * jitter * proc_factor[p]
    # komunikasi pakai CCR
    comm = np.zeros((n, n))
    for (u, v) in G.edges:
        comm[u, v] = ccr * np.mean(exec_time[u]) * rng.beta(2, 5)
    return exec_time, comm
