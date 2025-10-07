# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

def plot_by(group_key, metric):
    g = df.groupby([group_key, "algo"])[metric].mean().reset_index()
    algos = ["HEFT","GA"]
    xs = sorted(g[group_key].unique())
    width = 0.35
    x_idx = range(len(xs))

    for i, algo in enumerate(algos):
        ys = [g[(g[group_key]==x)&(g["algo"]==algo)][metric].values.mean() if not g[(g[group_key]==x)&(g["algo"]==algo)].empty else 0
              for x in xs]
        plt.figure()
        plt.bar([xi + (i-0.5)*width for xi in x_idx], ys, width=width)
        plt.title(f"{metric} vs {group_key} ({algo})")
        plt.xlabel(group_key); plt.ylabel(metric)
        plt.xticks(list(x_idx), xs, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{metric}_by_{group_key}_{algo}.png")
        plt.close()

for m in ["makespan","energy","cost","reliability","load_balance"]:
    plot_by("ccr", m)
    plot_by("n", m)

print("Saved charts: *_by_*.png")
