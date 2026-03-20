import networkx as nx
import numpy as np


def build_graph_from_edges(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


def mst_dfs_tour(G, start=0):
    """Return a DFS preorder of the maximum spanning tree as initial tour."""
    mst = nx.maximum_spanning_tree(G)
    tour = list(nx.dfs_preorder_nodes(mst, source=start))
    return list(tour)


def two_opt_improve(path, sim_mat, max_iters=5):
    """Perform 2-opt local search to improve path (maximize adjacent similarity)."""
    improved = True
    iters = 0
    n = len(path)
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                a, b, c, d = path[i-1], path[i], path[j], path[j+1]
                if sim_mat[a, c] + sim_mat[b, d] > sim_mat[a, b] + sim_mat[c, d]:
                    path[i:j+1] = reversed(path[i:j+1])
                    improved = True
        if not improved:
            break
    return path
