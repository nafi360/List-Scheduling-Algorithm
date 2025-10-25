import numpy as np


def compute_metrics_from_timeline(assign, start, finish, P,
                                  power_active, power_idle,
                                  price_per_core_hour, lambda_fail):
    makespan = float(np.nanmax(finish))
    active = np.zeros(P)
    for v in range(len(assign)):
        p = int(assign[v])
        active[p] += float(finish[v] - start[v])
    idle = makespan - active
    energy = float(np.sum(active * power_active + idle * power_idle))
    core_hours = active / 3600.0
    cost = float(np.sum(core_hours * price_per_core_hour))
    R = 1.0
    for v in range(len(assign)):
        p = int(assign[v])
        rt = float(finish[v] - start[v])
        R *= np.exp(-float(lambda_fail[p]) * rt)
    load_balance = float(np.std(active))
    return makespan, energy, cost, R, load_balance
