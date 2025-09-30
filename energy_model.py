# energy_model.py
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple

EDGE_POWER_W = 5.0    # watts per processing unit on edge
CLOUD_POWER_W = 20.0  # watts per processing unit on cloud

@dataclass
class Node:
    name: str
    speed: float
    net_latency: float
    busy_until: float = 0.0
    total_proc_time: float = 0.0
    processed_tasks: int = 0
    energy_consumed: float = 0.0

    def expected_finish_time(self, arrival_time: float, demand: float) -> float:
        start = max(arrival_time, self.busy_until)
        proc_time = demand / self.speed
        finish = start + proc_time + self.net_latency
        return finish

    def assign_task(self, arrival_time: float, demand: float, power_watt:float=10.0) -> Tuple[float, float]:
        start = max(arrival_time, self.busy_until)
        proc_time = demand / self.speed
        finish = start + proc_time + self.net_latency
        self.total_proc_time += proc_time
        self.busy_until = finish
        self.processed_tasks += 1
        self.energy_consumed += power_watt * proc_time
        return start, finish

@dataclass
class Task:
    id: int
    arrival: float
    demand: float

def generate_tasks(num_tasks: int, sim_time: float, demand_mean: float=50, demand_std: float=15, seed: int=None) -> List[Task]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    arrivals = np.sort(np.random.uniform(0, sim_time, size=num_tasks))
    demands = np.clip(np.random.normal(demand_mean, demand_std, size=num_tasks), 5, None)
    tasks = [Task(i, float(arrivals[i]), float(demands[i])) for i in range(num_tasks)]
    return tasks

def compute_jain_index(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 1.0
    return (arr.sum()**2) / (len(arr) * (arr**2).sum())

def compute_energy(nodes: List[Node]) -> float:
    return sum(n.energy_consumed for n in nodes)
