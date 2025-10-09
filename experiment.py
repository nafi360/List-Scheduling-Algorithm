import csv
import itertools
import random
from config import PROCESSORS, CCRs, SHAPES, MAX_OUTDEG_LIST, NODES, BETA, REPEATS, GA_PARAMS
from dag_gen import gen_dag, assign_costs, plot_dag
from heft import heft_schedule
from ga import ga_schedule
from metrics import compute_metrics_from_timeline


def run_once(n, ccr, shape, P, max_outdeg, seed, preview=False):
    G, _ = gen_dag(n, shape, max_outdeg=max_outdeg, seed=seed)
    exec_time, comm = assign_costs(G, P, ccr, beta=BETA, seed=seed)

    if preview and random.random() < 0.05:
        filename = f"dag_preview_n{n}_ccr{ccr}_shape{shape}_P{P}_rep{seed}.png"
        plot_dag(G, title=f"Preview DAG n={n}, CCR={ccr}, shape={shape}, P={P}", save_path=filename)

    assign_h, start_h, finish_h = heft_schedule(G, exec_time, comm)
    met_h = compute_metrics_from_timeline(assign_h, start_h, finish_h, P,
                                          GA_PARAMS.power_active[:P], GA_PARAMS.power_idle[:P],
                                          GA_PARAMS.price_per_core_hour, GA_PARAMS.lambda_fail[:P])

    _, _, assign_g, start_g, finish_g, met_g = ga_schedule(G, exec_time, comm, P, GA_PARAMS)

    return met_h, met_g


def main():
    rows = []
    seed_base = 123
    max_outdeg_list = MAX_OUTDEG_LIST if MAX_OUTDEG_LIST is not None else [None]
    for n, ccr, shape, P, outdeg in itertools.product(NODES, CCRs, SHAPES, PROCESSORS, max_outdeg_list):
        for rep in range(REPEATS):
            seed = seed_base + rep
            try:
                preview_flag = random.random() < 0.02
                met_h, met_g = run_once(n, ccr, shape, P, outdeg, seed, preview=preview_flag)
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
                print(f"OK n={n} ccr={ccr} shape={shape} P={P} rep={rep}")
            except Exception as e:
                print("Fail:", n, ccr, shape, P, rep, e)

    if rows:
        with open("results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Saved results.csv")
    else:
        print("Tidak ada hasil. Cek konfigurasi.")

if __name__ == "__main__":
    main()