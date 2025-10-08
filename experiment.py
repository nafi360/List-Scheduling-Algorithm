import csv
import itertools
from config import P, CCRs, SHAPES, OUTDEG, NODES, COMP_RANGE, REPEATS, GA_PARAMS
from dag_gen import gen_dag, assign_costs, plot_dag
from heft import heft_schedule
from ga import ga_schedule
from metrics import compute_metrics_from_timeline
import random

def run_once(n, ccr, shape, outdeg, seed, preview=False):
    G, _ = gen_dag(n, shape, max_outdeg=outdeg, seed=seed)
    exec_time, comm = assign_costs(G, P, ccr, comp_cost_range=COMP_RANGE, seed=seed)

    # Cetak DAG hanya untuk sebagian kecil kombinasi
    if preview and random.random() < 0.05:  # hanya 5% dari kasus dicetak
        filename = f"dag_preview_n{n}_ccr{ccr}_shape{shape}_out{outdeg}_rep{seed}.png"
        plot_dag(G, title=f"Preview DAG n={n}, CCR={ccr}, shape={shape}", save_path=filename)

    # HEFT
    assign_h, start_h, finish_h = heft_schedule(G, exec_time, comm)
    met_h = compute_metrics_from_timeline(assign_h, start_h, finish_h, P,
                                          GA_PARAMS.power_active, GA_PARAMS.power_idle,
                                          GA_PARAMS.price_per_core_hour, GA_PARAMS.lambda_fail)

    # GA
    _, _, assign_g, start_g, finish_g, met_g = ga_schedule(G, exec_time, comm, P, GA_PARAMS)

    return met_h, met_g


def main():
    rows = []
    seed_base = 123
    for n, ccr, shape, outdeg in itertools.product(NODES, CCRs, SHAPES, OUTDEG):
        for rep in range(REPEATS):
            seed = seed_base + rep
            try:
                # preview DAG hanya sebagian kecil saja
                preview_flag = random.random() < 0.02  # hanya 2% eksperimen yang cetak DAG
                met_h, met_g = run_once(n, ccr, shape, outdeg, seed, preview=preview_flag)

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

    if rows:
        with open("results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Saved results.csv")
    else:
        print("Tidak ada hasil. Cek konfigurasi.")

if __name__ == "__main__":
    main()