# experiment.py
import itertools, csv
import numpy as np
import networkx as nx
from dag_gen import gen_dag, assign_costs
from heft import heft_schedule
from ga import ga_schedule, evaluate

P = 8
CCRs = [0.1, 0.5, 0.8, 1, 2, 5, 8, 10]
SHAPES = [0.5, 0.8, 1, 2, 4]
OUTDEG = [1,2,3,4]
NODES = [10, 20, 30, 50, 80, 100]
COMP_RANGE = (0.5, 1.5)  # ganti sesuai “range percentage of computation cost yang lo kirim”

GA_PARAMS = dict(pop=40, gens=80, pc=0.9, pm=0.1,
                 weights=[0.4,0.2,0.1,0.2,0.1],
                 power_active=np.full(P, 55.0),
                 power_idle=np.full(P, 12.0),
                 price_per_core_hour=0.02,
                 lambda_fail=np.full(P, 1e-6))

def run_once(n, ccr, shape, outdeg, seed):
    G, levels = gen_dag(n, shape, max_outdeg=outdeg, seed=seed)
    exec_time, comm = assign_costs(G, P, ccr, comp_cost_range=COMP_RANGE, seed=seed)

    # HEFT
    assign_h, start_h, finish_h = heft_schedule(G, exec_time, comm)
    makespan_h = float(finish_h.max())
    # derive metrics like GA.evaluate untuk konsistensi
    from ga import evaluate
    met_h = evaluate(G, exec_time, comm,
                     list(nx.topological_sort(G)), assign_h,
                     GA_PARAMS['power_active'], GA_PARAMS['power_idle'],
                     GA_PARAMS['price_per_core_hour'], GA_PARAMS['lambda_fail'])

    # GA
    best_order, best_map, met_g = ga_schedule(G, exec_time, comm, P, GA_PARAMS)

    return met_h, met_g

def main():
    rows = []
    seed_base = 123
    for n, ccr, shape, outdeg in itertools.product(NODES, CCRs, SHAPES, OUTDEG):
        for rep in range(3):  # ulangan biar stabil
            seed = seed_base + rep
            try:
                met_h, met_g = run_once(n, ccr, shape, outdeg, seed)
                rows.append({
                    "n": n, "ccr": ccr, "shape": shape, "outdeg": outdeg, "rep": rep,
                    "algo": "HEFT",
                    "makespan": met_h[0], "energy": met_h[1], "cost": met_h[2],
                    "reliability": met_h[3], "load_balance": met_h[4]
                })
                rows.append({
                    "n": n, "ccr": ccr, "shape": shape, "outdeg": outdeg, "rep": rep,
                    "algo": "GA",
                    "makespan": met_g[0], "energy": met_g[1], "cost": met_g[2],
                    "reliability": met_g[3], "load_balance": met_g[4]
                })
                print(f"OK n={n} ccr={ccr} shape={shape} out={outdeg} rep={rep}")
            except Exception as e:
                print("Fail case:", n, ccr, shape, outdeg, rep, e)
    # simpan
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
