import csv
import itertools
from config import PROCESSORS, CCRs, SHAPES, MAX_OUTDEG_LIST, NODES, BETA, REPEATS, GA_PARAMS
from dag_gen import gen_dag, assign_costs
from heft import heft_schedule
from ga import ga_schedule
from runner import run_dag_realtime
from metrics import compute_metrics_from_timeline


def run_once_real(n, ccr, shape, P, max_outdeg, seed):
    G, _ = gen_dag(n, shape, max_outdeg=max_outdeg, seed=seed)
    exec_time, comm = assign_costs(G, P, ccr, beta=BETA, seed=seed)

    assign_h, _sh, _fh = heft_schedule(G, exec_time, comm)
    start_hr, finish_hr = run_dag_realtime(G, assign_h, exec_time, comm, P=P)
    met_h = compute_metrics_from_timeline(assign_h, start_hr, finish_hr, P,
                                          GA_PARAMS.power_active[:P], GA_PARAMS.power_idle[:P],
                                          GA_PARAMS.price_per_core_hour, GA_PARAMS.lambda_fail[:P])

    _, _, assign_g, _bs, _bf, _metg = ga_schedule(G, exec_time, comm, P, GA_PARAMS)
    start_gr, finish_gr = run_dag_realtime(G, assign_g, exec_time, comm, P=P)
    met_g = compute_metrics_from_timeline(assign_g, start_gr, finish_gr, P,
                                          GA_PARAMS.power_active[:P], GA_PARAMS.power_idle[:P],
                                          GA_PARAMS.price_per_core_hour, GA_PARAMS.lambda_fail[:P])
    return met_h, met_g


def main():
    rows = []
    seed_base = 777
    max_outdeg_list = MAX_OUTDEG_LIST if MAX_OUTDEG_LIST is not None else [None]
    for n, ccr, shape, P, outdeg in itertools.product(NODES, CCRs, SHAPES, PROCESSORS, max_outdeg_list):
        for rep in range(REPEATS):
            seed = seed_base + rep
            try:
                met_h, met_g = run_once_real(n, ccr, shape, P, outdeg, seed)
                rows.append({
                    "n": n, "ccr": ccr, "shape": shape, "P": P, "outdeg": (outdeg if outdeg is not None else -1), "rep": rep,
                    "algo": "HEFT",
                    "makespan": met_h[0], "energy": met_h[1], "cost": met_h[2],
                    "reliability": met_h[3], "load_balance": met_h[4]
                })
                rows.append({
                    "n": n, "ccr": ccr, "shape": shape, "P": P, "outdeg": (outdeg if outdeg is not None else -1), "rep": rep,
                    "algo": "GA",
                    "makespan": met_g[0], "energy": met_g[1], "cost": met_g[2],
                    "reliability": met_g[3], "load_balance": met_g[4]
                })
                print(f"REAL OK n={n} ccr={ccr} shape={shape} P={P} rep={rep}")
            except Exception as e:
                print("REAL Fail:", n, ccr, shape, P, rep, e)

    if rows:
        with open("results_real.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Saved results_real.csv")

if __name__ == "__main__":
    main()