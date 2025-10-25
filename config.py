from dataclasses import dataclass
import numpy as np

# Sesuai tabel param
NODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CCRs = [0.1, 0.5, 0.8, 1, 2, 5, 8, 10]
SHAPES = [0.5, 0.8, 1, 2, 4]
# Out-degree: gunakan rentang 1..v (dinamis). Jika ingin batasi, isi list di bawah.
MAX_OUTDEG_LIST = None  # None berarti 1..v; atau misal [1,2,3,4]

# Rentang eksekusi per-proc berdasarkan beta
# w_i * (1 - beta/2) <= w_{i,j} <= w_i * (1 + beta/2)
BETA = 0.5
REPEATS = 3


@dataclass(frozen=True)
class InstanceSpec:
    """Profil instansi AWS EC2 yang digunakan untuk eksperimen."""
    name: str
    vcpus: int
    memory_gib: float
    price_per_hour: float
    power_active_w: float
    power_idle_w: float
    failure_rate: float


INSTANCE_SPECS = [
    InstanceSpec("t3.small", 2, 2.0, 0.0264, 28.0, 8.5, 1.5e-6),
    InstanceSpec("m6a.xlarge", 4, 16.0, 0.2112, 65.0, 18.0, 1.2e-6),
    InstanceSpec("m6a.2xlarge", 8, 32.0, 0.4672, 120.0, 25.0, 1.1e-6),
]

PROCESSORS = [spec.vcpus for spec in INSTANCE_SPECS]
_INSTANCE_BY_VCPU = {spec.vcpus: spec for spec in INSTANCE_SPECS}


@dataclass(frozen=True)
class ResourceProfile:
    instance: InstanceSpec
    power_active: np.ndarray
    power_idle: np.ndarray
    price_per_core_hour: np.ndarray
    lambda_fail: np.ndarray


def get_resource_profile(vcpus: int) -> ResourceProfile:
    """Bangun profil sumber daya untuk jumlah vCPU tertentu."""
    if vcpus not in _INSTANCE_BY_VCPU:
        raise ValueError(f"Tidak ada profil instansi untuk {vcpus} vCPU")
    spec = _INSTANCE_BY_VCPU[vcpus]
    power_active = np.full(vcpus, spec.power_active_w, dtype=float)
    power_idle = np.full(vcpus, spec.power_idle_w, dtype=float)
    price_per_core_hour = np.full(vcpus, spec.price_per_hour / spec.vcpus, dtype=float)
    lambda_fail = np.full(vcpus, spec.failure_rate, dtype=float)
    return ResourceProfile(spec, power_active, power_idle, price_per_core_hour, lambda_fail)


@dataclass
class GAParams:
    pop: int = 40
    gens: int = 80
    pc: float = 0.9
    pm: float = 0.1
    weights: tuple = (0.4, 0.2, 0.1, 0.2, 0.1)


GA_PARAMS = GAParams()
