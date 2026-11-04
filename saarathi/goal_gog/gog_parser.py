from dataclasses import dataclass, field
from typing import List, Set, Tuple
import networkx as nx
import torch
import torch.nn.functional as F

@dataclass
class SubgoalNode:
    node_id: str
    goal_desc: str
    preconditions: Set[str]
    postconditions: Set[str]
    embedding: torch.Tensor = field(default_factory=lambda: torch.randn(128))

class GoalOrientedGraphPlanner:
    """
    Implements Section V-C (Eq. 18): G = (g, s, d, a) and GoG = (V_G, E_G).
    Merges equivalent nodes when cos(psi(g_i), psi(g_j)) >= tau_eq and builds DAG.
    """
    def __init__(self, tau_eq: float = 0.85):
        self.tau_eq = tau_eq
        self.graph = nx.DiGraph()

    def parse_request_to_tuple(self, refined_query: str, start_node: str, matched_sku: dict) -> Tuple[str, str, str, List[str]]:
        g = f"Acquire {matched_sku['name']}"
        s = start_node
        d = matched_sku["aisle"]

        nodes = [
            SubgoalNode("u1", f"Navigate from {s} to {d}", {"at_start"}, {"at_target_aisle"}),
            SubgoalNode("u2", f"Scan shelf in {d}", {"at_target_aisle"}, {"shelf_boxes_detected"}),
            SubgoalNode("u3", f"Disambiguate {matched_sku['name']} via CAQE", {"shelf_boxes_detected"}, {"sku_verified"}),
            SubgoalNode("u4", f"Announce price and allergens for {matched_sku['name']}", {"sku_verified"}, {"goal_complete"})
        ]

        self._build_gog(nodes)
        action_chain = [self.graph.nodes[n]["data"].goal_desc for n in nx.topological_sort(self.graph)]
        return (g, s, d, action_chain)

    def _build_gog(self, nodes: List[SubgoalNode]):
        self.graph.clear()
        merged_nodes = []
        for candidate in nodes:
            is_duplicate = False
            for existing in merged_nodes:
                sim = F.cosine_similarity(candidate.embedding.unsqueeze(0), existing.embedding.unsqueeze(0)).item()
                if sim >= self.tau_eq:
                    existing.preconditions.update(candidate.preconditions)
                    existing.postconditions.update(candidate.postconditions)
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged_nodes.append(candidate)
                self.graph.add_node(candidate.node_id, data=candidate)

        # Instantiate directed edge (u_i -> u_j) iff P_post(u_i) intersect P_pre(u_j) != empty
        for u_i in merged_nodes:
            for u_j in merged_nodes:
                if u_i.node_id != u_j.node_id:
                    if u_i.postconditions.intersection(u_j.preconditions):
                        self.graph.add_edge(u_i.node_id, u_j.node_id)