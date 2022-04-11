import networkx as nx
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator


def graph_to_ndarray(graph):
    out_nodes = np.empty((1, 3))
    out_edges = np.empty((1, 3))
    visited_nodes = list()
    visited_node_pairs = list()
    for (start_node, end_node) in graph.edges():
        if (not (start_node, end_node) in visited_node_pairs) and (not (end_node, start_node) in visited_node_pairs):
            edge = graph[start_node][end_node]['pts']
            out_edges = np.vstack([out_edges, edge])

            # Avoid duplicates
            if not (start_node in visited_nodes):
                out_nodes = np.vstack([out_nodes, graph.nodes[start_node]['o']])
                visited_nodes.append(start_node)
            if not (end_node in visited_nodes):
                out_nodes = np.vstack([out_nodes, graph.nodes[end_node]['o']])
                visited_nodes.append(end_node)

            visited_node_pairs.append((start_node, end_node))

    return np.vstack([out_edges, out_nodes]), out_nodes, out_edges


def get_bifurcation_nodes(graph: nx.Graph):
    # Vertex degree relates to the number of branches connected to a given node
    out_nodes = np.empty((1, 3))
    bif_nodes_id = list()
    for node_num, deg in graph.degree:
        if deg > 1:
            bif_nodes_id.append(node_num)
            out_nodes = np.vstack([out_nodes, graph.nodes[node_num]['o']])

    return out_nodes, bif_nodes_id


def apply_displacement(pts_list: np.ndarray, interpolator: [RegularGridInterpolator, LinearNDInterpolator]):
    pts_list = pts_list.astype(np.float)
    ret_val = pts_list + interpolator(pts_list).squeeze()
    return ret_val


def deform_graph(graph, dm_interpolator: [RegularGridInterpolator, LinearNDInterpolator]):
    def_graph = nx.Graph()
    for (start_node, end_node) in graph.edges():
        edge = graph[start_node][end_node]['pts']
        def_edge = apply_displacement(edge, dm_interpolator)

        def_start_node_pts = apply_displacement(graph.nodes[start_node]['pts'], dm_interpolator)
        def_end_node_pts = apply_displacement(graph.nodes[end_node]['pts'], dm_interpolator)

        def_start_node_o = apply_displacement(graph.nodes[start_node]['o'], dm_interpolator)
        def_end_node_o = apply_displacement(graph.nodes[end_node]['o'], dm_interpolator)

        def_graph.add_node(start_node, pts=def_start_node_pts, o=def_start_node_o)
        def_graph.add_node(end_node, pts=def_end_node_pts, o=def_end_node_o)
        def_graph.add_edge(start_node, end_node, pts=def_edge, weight=len(def_edge))
    return def_graph


def subsample_graph(graph: nx.Graph, num_samples=3):
    sub_graph = nx.Graph()
    for (start_node, end_node) in graph.edges():
        edge = graph[start_node][end_node]['pts']
        edge_len = edge.shape[0]
        sub_edge_len = (edge_len - 2) // num_samples # Do not count the pts corresponding to the nodes (-2)

        sub_edge = [edge[0]]
        include_last = bool((edge_len - 2) % num_samples)  # Skip the last point, as this is too close to the node
        if sub_edge_len:
            idxs = np.arange(0, edge_len, num_samples)[1:] if include_last else np.arange(0, edge_len, num_samples)[1:-1]
            for i in idxs:
                sub_edge.append(edge[i])

        sub_edge.append(edge[-1])
        sub_edge = np.asarray(sub_edge)
        sub_graph.add_node(start_node, pts=graph.nodes[start_node]['pts'], o=graph.nodes[start_node]['o'])
        sub_graph.add_node(end_node, pts=graph.nodes[end_node]['pts'], o=graph.nodes[end_node]['o'])
        sub_graph.add_edge(start_node, end_node, pts=sub_edge, weight=len(sub_edge))
    return sub_graph

