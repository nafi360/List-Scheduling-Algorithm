import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

_rng = np.random.default_rng()

def gen_dag(n_nodes: int, shape: float, max_outdeg: int, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    # jumlah level dipengaruhi shape
    n_levels = max(2, int(np.ceil(np.sqrt(n_nodes) / max(0.5, min(shape, 4)))))
    levels = [[] for _ in range(n_levels)]

    for v in range(n_nodes):
        lv = int(_rng.normal(loc=n_levels / 2, scale=max(1, int(n_levels / (2 * max(0.5, shape))))))
        lv = max(0, min(n_levels - 1, lv))
        levels[lv].append(v)

    if not levels[0]:
        levels[0].append(0)
    if not levels[-1]:
        levels[-1].append(n_nodes - 1)

    for i in range(n_levels - 1):
        for u in levels[i]:
            outdeg = int(_rng.integers(0, max_outdeg + 1))
            used = set()
            for _ in range(outdeg):
                j = int(_rng.integers(i + 1, n_levels))
                if levels[j]:
                    v = int(_rng.choice(levels[j]))
                    if v != u and (u, v) not in used:
                        G.add_edge(u, v)
                        used.add((u, v))

    assert nx.is_directed_acyclic_graph(G)

    # pastikan ada jembatan antar level bertetangga
    for i in range(n_levels - 1):
        if not any((u, v) in G.edges for u in levels[i] for v in levels[i + 1]):
            u = int(_rng.choice(levels[i]))
            v = int(_rng.choice(levels[i + 1]))
            G.add_edge(u, v)

    return G, levels


def assign_costs(G: nx.DiGraph, P: int, ccr: float, comp_cost_range=(0.5, 1.5), seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    n = G.number_of_nodes()
    base = _rng.uniform(1.0, 10.0, size=n)
    proc_factor = _rng.uniform(0.7, 1.3, size=P)
    exec_time = np.zeros((n, P))
    lo, hi = comp_cost_range
    jitter = _rng.uniform(lo, hi, size=n)
    for v in range(n):
        for p in range(P):
            exec_time[v, p] = base[v] * jitter[v] * proc_factor[p]

    comm = np.zeros((n, n))
    for (u, v) in G.edges:
        comm[u, v] = ccr * float(np.mean(exec_time[u])) * _rng.beta(2, 5)
    return exec_time, comm


def _try_graphviz_layout(G):
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog="dot")
    except Exception:
        return nx.spring_layout(G, seed=42)


def plot_dag(G: nx.DiGraph, title: str = "Random DAG", save_path: str | None = None):
    pos = _try_graphviz_layout(G)
    plt.figure(figsize=(6.5, 4.0))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="#87CEEB",
            arrowsize=14, font_size=8, font_weight="bold", edge_color="#777777")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()