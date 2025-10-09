import csv
import itertools
import numpy as np
from config import P, CCRs, SHAPES, OUTDEG, NODES, COMP_RANGE, REPEATS, GA_PARAMS
from dag_gen import gen_dag, assign_costs
from heft import heft_schedule
from ga import ga_schedule
from metrics import compute_metrics_from_timeline
from runner import execute_schedule


def run_once_real(G, exec_time, comm, assign, label):
    # Eksekusi nyata; start/finish dalam detik relatif dari t0
    start, finish = execute_schedule(G, assign, exec_time, comm, processes=P)
    met = compute_metrics_from_timeline(assign, start, finish, P,
                                        GA_PARAMS.power_active, GA_PARAMS.power_idle,
                                        GA_PARAMS.price_per_core_hour, GA_PARAMS.lambda_fail)
    print(f"Executed real DAG for {label}: makespan={met[0]:.3f}s")
    return met


def main():
    rows = []
    seed_base = 999
    # Demi biaya, default contoh kecil; ubah ke NODES penuh kalau mau berat
    for n, ccr, shape, outdeg in itertools.product([10, 30, 50], [0.5, 1, 5], [1, 2], [2]):
        for rep in range(1):
            seed = seed_base + rep
            G, _ = gen_dag(n, shape, max_outdeg=outdeg, seed=seed)
            exec_time, comm = assign_costs(G, P, ccr, comp_cost_range=COMP_RANGE, seed=seed)

            # HEFT mapping
            assign_h, start_h_s, finish_h_s = heft_schedule(G, exec_time, comm)
            met_h = run_once_real(G, exec_time, comm, assign_h, label="HEFT")

            # GA mapping
            _, _, assign_g, _, _, _ = ga_schedule(G, exec_time, comm, P, GA_PARAMS)
            met_g = run_once_real(G, exec_time, comm, assign_g, label="GA")

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

    if rows:
        with open("results_real.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Saved results_real.csv")

if __name__ == "__main__":
    main()