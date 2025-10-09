import argparse
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["makespan","energy","cost","reliability","load_balance"]


def plot_by(df, group_key: str, metric: str, suffix: str = ""):
    g = df.groupby([group_key, "algo"], as_index=False)[metric].mean()
    if g.empty:
        return
    algos = ["HEFT", "GA"]
    xs = sorted(g[group_key].unique())
    plt.figure()
    for algo in algos:
        ys = [g[(g[group_key]==x) & (g["algo"]==algo)][metric].values.mean() if not g[(g[group_key]==x) & (g["algo"]==algo)].empty else 0 for x in xs]
        plt.plot(xs, ys, marker='o', label=algo)
    plt.title(f"{metric} vs {group_key}{' ('+suffix+')' if suffix else ''}")
    plt.xlabel(group_key); plt.ylabel(metric)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out = f"{metric}_by_{group_key}_line{('_'+suffix) if suffix else ''}.png"
    plt.savefig(out, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results.csv", help="CSV hasil: results.csv atau results_real.csv")
    ap.add_argument("--suffix", default="", help="tambahan nama file output, misal 'real' atau 'sim'")
    df = pd.read_csv(ap.parse_args().file)
    suffix = ap.parse_args().suffix

    keys = [k for k in ["ccr","n","P"] if k in df.columns]
    for m in METRICS:
        for key in keys:
            plot_by(df, key, m, suffix=suffix)
    print("Line charts saved: *_line*.png")

if __name__ == "__main__":
    main()
