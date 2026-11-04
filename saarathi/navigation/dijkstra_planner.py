import networkx as nx
from typing import List, Tuple

class StoreNavigationGraph:
    """
    Implements Section V-D (Eqs. 19-20):
    G_m = (V, E) and pi* = argmin_pi sum c(v_i, v_{i+1}) via Dijkstra's algorithm.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self._init_default_supermarket()

    def _init_default_supermarket(self):
        edges = [
            ("Entrance", "Aisle_1", 4.0),
            ("Aisle_1", "Aisle_2", 3.5),
            ("Aisle_2", "Aisle_3", 3.5),
            ("Aisle_1", "Aisle_4", 5.0),
            ("Aisle_2", "Aisle_4", 4.0),
            ("Aisle_3", "Checkout", 4.5),
            ("Aisle_4", "Checkout", 3.0),
        ]
        for u, v, cost in edges:
            self.graph.add_edge(u, v, weight=cost)

    def shortest_route(self, start_node: str, dest_node: str) -> Tuple[List[str], float]:
        path = nx.shortest_path(self.graph, source=start_node, target=dest_node, weight="weight")
        cost = nx.shortest_path_length(self.graph, source=start_node, target=dest_node, weight="weight")
        return path, float(cost)

    def mark_edge_blocked(self, u: str, v: str):
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] = 1e6