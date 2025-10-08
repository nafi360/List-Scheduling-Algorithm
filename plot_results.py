import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")


def plot_by(group_key: str, metric: str):
    g = df.groupby([group_key, "algo"], as_index=False)[metric].mean()
    algos = ["HEFT", "GA"]
    xs = sorted(g[group_key].unique())

    plt.figure()
    for algo in algos:
        ys = [
            g[(g[group_key] == x) & (g["algo"] == algo)][metric].values.mean()
            if not g[(g[group_key] == x) & (g["algo"] == algo)].empty else 0
            for x in xs
        ]
        plt.plot(xs, ys, marker='o', label=algo)

    plt.title(f"{metric} vs {group_key}")
    plt.xlabel(group_key)
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{metric}_by_{group_key}_line.png", dpi=300)
    plt.close()


for m in ["makespan", "energy", "cost", "reliability", "load_balance"]:
    plot_by("ccr", m)
    plot_by("n", m)

print("Line charts saved: *_line.png")
