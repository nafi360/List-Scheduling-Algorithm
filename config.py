from dataclasses import dataclass
import numpy as np

# Sesuai tabel param
NODES = [10,20,30,40,50,60,70,80,90,100]
CCRs = [0.1, 0.5, 0.8, 1, 2, 5, 8, 10]
PROCESSORS = [4, 8, 16, 32]  # untuk real-run akan dibatasi oleh core fisik
SHAPES = [0.5, 0.8, 1, 2, 4]
# Out-degree: gunakan rentang 1..v (dinamis). Jika ingin batasi, isi list di bawah.
MAX_OUTDEG_LIST = None  # None berarti 1..v; atau misal [1,2,3,4]

# Rentang eksekusi per-proc berdasarkan beta
# w_i * (1 - beta/2) <= w_{i,j} <= w_i * (1 + beta/2)
BETA = 0.5
REPEATS = 3

@dataclass
class GAParams:
    pop: int = 40
    gens: int = 80
    pc: float = 0.9
    pm: float = 0.1
    weights: tuple = (0.4, 0.2, 0.1, 0.2, 0.1)
    # Panjang vector dibuat >= max(PROCESSORS)
    power_active: np.ndarray = np.full(32, 55.0)
    power_idle: np.ndarray = np.full(32, 12.0)
    price_per_core_hour: float = 0.02
    lambda_fail: np.ndarray = np.full(32, 1e-6)

GA_PARAMS = GAParams()