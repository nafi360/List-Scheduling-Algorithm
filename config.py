from dataclasses import dataclass
import numpy as np

P = 8  # jumlah prosesor

CCRs = [0.1, 0.5, 0.8, 1, 2, 5, 8, 10]
SHAPES = [0.5, 0.8, 1, 2, 4]
OUTDEG = [1, 2, 3, 4]
NODES = [10, 20, 30, 50, 80, 100]
COMP_RANGE = (0.5, 1.5)  # ganti sesuai kebutuhan
REPEATS = 3

@dataclass
class GAParams:
    pop: int = 40
    gens: int = 80
    pc: float = 0.9
    pm: float = 0.1
    weights: tuple = (0.4, 0.2, 0.1, 0.2, 0.1)  # [mksp, energy, cost, reliab, lb]
    power_active: np.ndarray = np.full(P, 55.0)  # Watt
    power_idle: np.ndarray = np.full(P, 12.0)    # Watt
    price_per_core_hour: float = 0.02            # USD/core-hour
    lambda_fail: np.ndarray = np.full(P, 1e-6)   # per detik

GA_PARAMS = GAParams()