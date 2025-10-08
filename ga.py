import numpy as np
import networkx as nx
from metrics import compute_metrics_from_timeline

_rng = np.random.default_rng()

def random_topological_order(G: nx.DiGraph):
    indeg = {v: G.in_degree(v) for v in G.nodes}
    zero = [v for v, d in indeg.items() if d == 0]
    order = []
    while zero:
        v = int(_rng.choice(zero))
        zero.remove(v)
        order.append(v)
        for w in G.successors(v):
            indeg[w] -= 1
            if indeg[w] == 0:
                zero.append(w)
    if len(order) != G.number_of_nodes():
        raise ValueError("Graph bukan DAG atau ada masalah topologi")
    return order


def repair_priority(G: nx.DiGraph, order):
    pos = {v: i for i, v in enumerate(order)}
    changed = True
    while changed:
        changed = False
        for u, v in G.edges:
            if pos[u] > pos[v]:
                iu, iv = pos[u], pos[v]
                order[iu], order[iv] = order[iv], order[iu]
                pos[order[iu]], pos[order[iv]] = iu, iv
                changed = True
    return order


def decode_schedule(G: nx.DiGraph, order, mapping, exec_time, comm):
    n, P = exec_time.shape
    avail = np.zeros(P)
    start = np.zeros(n)
    finish = np.zeros(n)
    assign = -np.ones(n, dtype=int)
    for v in order:
        p = int(mapping[v])
        ready = float(avail[p])
        for u in G.predecessors(v):
            cd = 0.0 if assign[u] == p else float(comm[u, v])
            ready = max(ready, float(finish[u]) + cd)
        start[v] = ready
        finish[v] = ready + float(exec_time[v, p])
        assign[v] = p
        avail[p] = float(finish[v])
    return assign, start, finish


def init_population(G, P, n, pop_size):
    pop = []
    for _ in range(pop_size):
        order = random_topological_order(G)
        _rng.shuffle(order)
        order = repair_priority(G, order)
        mapping = _rng.integers(0, P, size=n)
        pop.append((order, mapping))
    return pop


def crossover_order(o1, o2):
    n = len(o1)
    a, b = sorted(_rng.integers(0, n, size=2))
    child = [-1] * n
    child[a:b] = o1[a:b]
    fill = [x for x in o2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


def mutate_order(G, order, pm):
    order = order.copy()
    n = len(order)
    swaps = max(1, int(pm * n))
    for _ in range(swaps):
        i, j = _rng.integers(0, n, size=2)
        order[i], order[j] = order[j], order[i]
    return repair_priority(G, order)


def mutate_mapping(mapping, P, pm):
    m = mapping.copy()
    n = len(m)
    for i in range(n):
        if _rng.random() < pm:
            m[i] = int(_rng.integers(0, P))
    return m


def normalize_metrics(metrics_list):
    arr = np.array(metrics_list, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    norm = (arr - mins) / denom
    norm[:, 3] = 1.0 - norm[:, 3]  # reliability jadi 1-R
    return norm


def ga_schedule(G, exec_time, comm, P, params):
    n = exec_time.shape[0]
    pop = init_population(G, P, n, params.pop)
    hall = None
    hall_fit = float("inf")

    for _ in range(params.gens):
        raw_metrics = []
        decoded = []
        for (order, mapping) in pop:
            assign, start, finish = decode_schedule(G, order, mapping, exec_time, comm)
            met = compute_metrics_from_timeline(assign, start, finish, P,
                                                params.power_active, params.power_idle,
                                                params.price_per_core_hour, params.lambda_fail)
            raw_metrics.append(met)
            decoded.append((order, mapping, assign, start, finish))

        norm = normalize_metrics(raw_metrics)
        w = np.array(params.weights)
        fitness = (norm * w).sum(axis=1)

        idx = int(np.argmin(fitness))
        elite = pop[idx]
        elite_fit = float(fitness[idx])
        if elite_fit < hall_fit:
            hall_fit = elite_fit
            hall = (decoded[idx], raw_metrics[idx])

        def pick():
            a, b = _rng.integers(0, len(pop), size=2)
            return pop[a] if fitness[a] < fitness[b] else pop[b]

        newpop = [elite]
        while len(newpop) < params.pop:
            p1 = pick(); p2 = pick()
            o1, m1 = p1
            o2, m2 = p2
            if _rng.random() < params.pc:
                c_order = crossover_order(o1, o2)
                c_map = np.where(_rng.random(len(m1)) < 0.5, m1, m2)
            else:
                c_order, c_map = o1[:], m1.copy()
            c_order = mutate_order(G, c_order, params.pm)
            c_map = mutate_mapping(c_map, P, params.pm)
            newpop.append((c_order, c_map))
        pop = newpop[:params.pop]

    (best_order, best_map, best_assign, best_start, best_finish), best_met = hall
    return best_order, best_map, best_assign, best_start, best_finish, best_met