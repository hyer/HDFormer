from typing import Tuple, List
import numpy as np
import sys
sys.path.insert(0, "./")

def normalize_incidence_matrix(im: np.ndarray) -> np.ndarray:
    Dl = im.sum(-1)
    num_node = im.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    res = Dn @ im
    return res


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    source_graph = np.zeros((num_nodes, len(edges)), dtype='float32')
    target_graph = np.zeros((num_nodes, len(edges)), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    source_graph = normalize_incidence_matrix(source_graph)
    target_graph = normalize_incidence_matrix(target_graph)
    return source_graph, target_graph


class DiGraph():
    def __init__(self, skeleton):
        super().__init__()
        self.num_nodes = len(skeleton.parents())
        self.directed_edges_hop1 = [(parrent, child) for child, parrent in enumerate(skeleton.parents()) if parrent >= 0]
        self.directed_edges_hop2 = [(0,1,2),(0,4,5),(0,7,8),(1,2,3),(4,5,6),(7,8,9),(7,8,11),(7,8,14),(8,9,10),(8,11,12),(8,14,15),(11,12,13),(14,15,16)] # (parrent, child)
        self.directed_edges_hop3 = [(0,1,2,3),(0,4,5,6),(0,7,8,9),(7,8,9,10),(7,8,11,12),(7,8,14,15),(8,11,12,13),(8,14,15,16)]
        self.directed_edges_hop4 = [(0,7,8,9,10),(0,7,8,11,12),(0,7,8,14,15),(7,8,11,12,13),(7,8,14,15,16)]

        self.num_edges = len(self.directed_edges_hop1)
        self.edge_left = [0, 1, 2, 10, 11, 12]
        self.edge_right = [3, 4, 5, 13, 14, 15]
        self.edge_middle = [6, 7, 8, 9]
        self.center = 0  # for h36m data skeleton
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.directed_edges_hop1)


if __name__ == "__main__":
    from dataset.lib import get_skeleton

    skeleton = get_skeleton()
    graph = DiGraph(skeleton)
    source_M = graph.source_M
    target_M = graph.target_M
    print(source_M)
    print(target_M)
