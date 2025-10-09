from __future__ import annotations
import numpy as np

def task_matmul(n: int, reps: int = 1) -> None:
    for _ in range(reps):
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)
        _ = a @ b

def task_fft(n: int, reps: int = 10) -> None:
    x = np.random.rand(n)
    for _ in range(reps):
        _ = np.fft.fft(x)

def task_conv1d(n: int, k: int = 9, reps: int = 200) -> None:
    x = np.random.rand(n)
    kernel = np.random.rand(k)
    for _ in range(reps):
        _ = np.convolve(x, kernel, mode="valid")

WORKLOAD_FUNCS = {
    "matmul": task_matmul,
    "fft": task_fft,
    "conv1d": task_conv1d,
}

TASK_SPEC: dict[int, dict] = {}

def run_task(task_id: int) -> None:
    spec = TASK_SPEC.get(task_id)
    if not spec:
        task_matmul(256, reps=1)
        return
    fn = WORKLOAD_FUNCS[spec["type"]]
    fn(**spec.get("args", {}))