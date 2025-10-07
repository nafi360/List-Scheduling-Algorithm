# ga.py
import numpy as np
import networkx as nx
rng = np.random.default_rng()

def random_topo(G):
    # random topological order
    return list(nx.algorithms.dag.random_reference(G, seed=rng).topological_sort())

def repair_priority(G, order):
    # ensure parents before child using Kahn-like repair
    pos = {v:i for i,v in enumerate(order)}
    changed = True
    while changed:
        changed = False
        for u,v in G.edges:
            if pos[u] > pos[v]:
                # swap positions
                iu, iv = pos[u], pos[v]
                order[iu], order[iv] = order[iv], order[iu]
                pos[order[iu]], pos[order[iv]] = iu, iv
                changed = True
    return order

def decode_schedule(G, order, mapping, exec_time, comm):
    n, P = exec_time.shape
    avail = np.zeros(P)
    start = np.zeros(n)
    finish = np.zeros(n)
    assign = -np.ones(n, dtype=int)
    for v in order:
        p = mapping[v]
        ready = avail[p]
        for u in G.predecessors(v):
            cd = 0 if assign[u] == p else comm[u, v]
            ready = max(ready, finish[u] + cd)
        start[v] = ready
        finish[v] = ready + exec_time[v, p]
        assign[v] = p
        avail[p] = finish[v]
    return assign, start, finish

def init_population(G, P, n, pop_size):
    pop = []
    for _ in range(pop_size):
        order = list(nx.topological_sort(G))
        rng.shuffle(order)
        order = repair_priority(G, order)
        mapping = rng.integers(0, P, size=n)
        pop.append((order, mapping))
    return pop

def crossover_order(o1, o2):
    # order crossover (OX) preserving relative order
    n = len(o1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [-1]*n
    child[a:b] = o1[a:b]
    fill = [x for x in o2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]; idx += 1
    return child

def mutate_order(G, order, pm):
    order = order.copy()
    n = len(order)
    for _ in range(int(pm*n)):
        i, j = rng.integers(0, n, size=2)
        order[i], order[j] = order[j], order[i]
    return repair_priority(G, order)

def mutate_mapping(mapping, P, pm):
    m = mapping.copy()
    n = len(m)
    for i in range(n):
        if rng.random() < pm:
            m[i] = rng.integers(0, P)
    return m

def evaluate(G, exec_time, comm, order, mapping, power_active, power_idle, price_per_core_hour, lambda_fail):
    # decode
    assign, start, finish = decode_schedule(G, order, mapping, exec_time, comm)
    makespan = float(finish.max())
    # energy: sum over processors of (active_time*P_active + idle_time*P_idle)
    P = exec_time.shape[1]
    # timeline per proc
    active = np.zeros(P)
    for v in G.nodes:
        p = assign[v]
        active[p] += (finish[v] - start[v])
    idle = makespan - active  # asumsi tidak sleep
    energy = float((active*power_active + idle*power_idle).sum())
    # cost: proportional to core-time
    core_hours = active.sum() / 3600.0
    cost = float(core_hours * price_per_core_hour)
    # reliability: asumsikan failure rate per proc lambda_fail (per unit waktu)
    # R = product over tasks exp(-lambda_p * runtime_task)
    R = 1.0
    for v in G.nodes:
        p = assign[v]
        rt = finish[v]-start[v]
        R *= np.exp(-lambda_fail[p]*rt)
    # load balance: std dev dari active time antar prosesor
    lb = float(np.std(active))
    return makespan, energy, cost, R, lb

def normalize_metrics(metrics_list):
    # z-norm or min-max; pakai min-max simple
    arr = np.array(metrics_list)  # shape [k,5]
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    denom = np.where(maxs>mins, maxs-mins, 1.0)
    norm = (arr - mins)/denom
    # untuk reliability kita ingin “1-R”, supaya minim
    norm[:,3] = 1.0 - norm[:,3]
    return norm

def ga_schedule(G, exec_time, comm, P, params):
    n = exec_time.shape[0]
    pop = init_population(G, P, n, params['pop'])
    hall = None; hall_fit = float('inf')

    # constants for evaluation
    power_active = params.get('power_active', np.full(P, 50.0)) # Watt
    power_idle = params.get('power_idle', np.full(P, 10.0))
    price_per_core_hour = params.get('price_per_core_hour', 0.02) # USD/core-hour (isi sendiri)
    lambda_fail = params.get('lambda_fail', np.full(P, 1e-6))     # per detik

    for gen in range(params['gens']):
        # evaluate all
        raw_metrics = []
        decoded = []
        for (order, mapping) in pop:
            met = evaluate(G, exec_time, comm, order, mapping,
                           power_active, power_idle, price_per_core_hour, lambda_fail)
            raw_metrics.append(met)
            decoded.append((order, mapping, met))
        norm = normalize_metrics(raw_metrics)
        # scalarize
        w = np.array(params.get('weights',[0.4,0.2,0.1,0.2,0.1]))
        fitness = (norm * w).sum(axis=1)

        # elitism
        idx = int(np.argmin(fitness))
        elite = pop[idx]
        elite_met = raw_metrics[idx]
        if fitness[idx] < hall_fit:
            hall_fit = float(fitness[idx]); hall = elite, elite_met

        # selection (tournament)
        def pick():
            a, b = rng.integers(0, len(pop), size=2)
            return pop[a] if fitness[a] < fitness[b] else pop[b]

        newpop = [elite]
        while len(newpop) < params['pop']:
            p1 = pick(); p2 = pick()
            o1, m1 = p1
            o2, m2 = p2
            if rng.random() < params['pc']:
                c_order = crossover_order(o1, o2)
                c_map = np.where(rng.random(len(m1))<0.5, m1, m2)
            else:
                c_order, c_map = o1[:], m1.copy()
            c_order = mutate_order(G, c_order, params['pm'])
            c_map = mutate_mapping(c_map, P, params['pm'])
            newpop.append((c_order, c_map))
        pop = newpop[:params['pop']]
    # return best
    (best_order, best_map), best_met = hall
    return best_order, best_map, best_met
