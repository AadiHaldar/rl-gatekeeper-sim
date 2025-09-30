# protocols.py
import random

def proto_random(task, edge_nodes, cloud):
    return random.choice(edge_nodes + [cloud])

def proto_greedy(task, edge_nodes, cloud):
    all_nodes = edge_nodes + [cloud]
    best = min(all_nodes, key=lambda n: n.expected_finish_time(task.arrival, task.demand))
    return best

def proto_edge_preferred(task, edge_nodes, cloud):
    best_edge = min(edge_nodes, key=lambda n: n.expected_finish_time(task.arrival, task.demand))
    if best_edge.expected_finish_time(task.arrival, task.demand) <= cloud.expected_finish_time(task.arrival, task.demand):
        return best_edge
    return cloud

def proto_auction(task, edge_nodes, cloud):
    all_nodes = edge_nodes + [cloud]
    scores = {n.name: -n.expected_finish_time(task.arrival, task.demand) for n in all_nodes}
    best_name = max(scores, key=scores.get)
    best_node = next(n for n in all_nodes if n.name == best_name)
    return best_node

def proto_nash(task, edge_nodes, cloud):
    return proto_greedy(task, edge_nodes, cloud)
