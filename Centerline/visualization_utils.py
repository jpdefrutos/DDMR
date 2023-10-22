import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from ddmr.utils.visualization import add_axes_arrows_3d, remove_tick_labels, set_axes_size
import os


def _plot_graph(graph, ax, nodes_colour='C3', edges_colour='C1', plot_nodes=True, plot_edges=True, add_axes=True):
    if plot_edges:
        for (start_node, end_node) in graph.edges():
            edge_pts = graph[start_node][end_node]['pts']
            edge_pts = np.vstack([graph.nodes[start_node]['o'], edge_pts])
            edge_pts = np.vstack([edge_pts, graph.nodes[end_node]['o']])
            ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], edges_colour)

    if plot_nodes:
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        if len(ps.shape) > 1:
            ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], nodes_colour)
        else:
            ax.scatter(ps[0], ps[1], ps[2], nodes_colour)
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 63)
    ax.set_zlim(0, 63)
    remove_tick_labels(ax, True)
    if add_axes:
        add_axes_arrows_3d(ax, x_color='r', y_color='g', z_color='b')
    ax.view_init(None, 45)

    return ax


def plot_skeleton(img, skeleton, graph, filename='skeleton', extension=['.png']):
    if not isinstance(extension, list):
        extension = [extension]
    # Skeleton
    f = plt.figure(figsize=(5, 5))
    ax = f.add_subplot(111, projection='3d')

    coords = np.argwhere(skeleton)
    i = coords[:, 0]
    j = coords[:, 1]
    k = coords[:, 2]

    seg = ax.voxels(img, facecolors=(0., 0., 1., 0.3), label='image')
    ske = ax.scatter(i, j, k, color='C1', label='skeleton', s=1)
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 63)
    ax.set_zlim(0, 63)
    remove_tick_labels(ax, True)
    add_axes_arrows_3d(ax, x_color='r', y_color='g', z_color='b')
    ax.view_init(None, 45)
    for ex in extension:
        f.savefig(filename + '_segmentation_skeleton' + ex)

    # Combined
    ax = _plot_graph(graph, ax, 'r', 'r')

    for ex in extension:
        f.savefig(filename + '_combined' + ex)
    plt.close()

    # Graph
    f = plt.figure(figsize=(5, 5))
    ax = f.add_subplot(111, projection='3d')

    ax = _plot_graph(graph, ax)

    for ex in extension:
        f.savefig(filename + '_graph' + ex)
    plt.close()




def compare_graphs(graph_0, graph_1, graph_names=None, filename='compare_graphs'):
    f = plt.figure(figsize=(12, 5))
    if graph_names is None:
        graph_names =['graph_0', 'graph_1']
    else:
        assert len(graph_names) == 2, 'A different name is expected for each graph'
    ax = f.add_subplot(131, projection='3d')
    ax = _plot_graph(graph_0, ax)
    ax.set_title(graph_names[0], y=-0.01)

    ax = f.add_subplot(132, projection='3d')
    ax = _plot_graph(graph_1, ax)
    ax.set_title(graph_names[1])

    ax = f.add_subplot(133, projection='3d')
    ax = _plot_graph(graph_0, ax, 'C2', 'C2', plot_nodes=False)
    ax = _plot_graph(graph_1, ax, 'C4', 'C4', plot_nodes=False)
    legend_elements = [Line2D([0], [0], color='C2', lw=2, label=graph_names[0]),
                       Line2D([0], [0], color='C4', lw=2, label=graph_names[1])]
    ax.legend(handles=legend_elements)

    f.savefig(filename + '_compare_graphs.png')
    plt.close()


def plot_cpd_registration_step(iteration, error, X, Y, out_folder, add_axes=True, pdf=True):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, .9, .9], projection='3d')
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='C1', label='Fixed')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='C2', label='Moving')

    ax.text2D(0.95, 0.98, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    #ax.text2D(0.95, 0.90, 'Error: {:10.4f}'.format(
    #    error), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')

    if add_axes:
        x_range = [np.min(np.hstack([X[:, 0], Y[:, 0]])), np.max(np.hstack([X[:, 0], Y[:, 0]]))]
        y_range = [np.min(np.hstack([X[:, 1], Y[:, 1]])), np.max(np.hstack([X[:, 1], Y[:, 1]]))]
        z_range = [np.min(np.hstack([X[:, 2], Y[:, 2]])), np.max(np.hstack([X[:, 2], Y[:, 2]]))]
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_range[0], z_range[1])

        remove_tick_labels(ax, True)
        add_axes_arrows_3d(ax, x_color='r', y_color='g', z_color='b', arrow_length=25, dist_arrow_text=3)
    ax.view_init(None, 45)

    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(os.path.join(out_folder, '{:04d}.png'.format(iteration)))
    if pdf:
        fig.savefig(os.path.join(out_folder, '{:04d}.pdf'.format(iteration)))
    plt.close()


def plot_cpd(fix_pts, mov_pts, fix_centroid, mov_centroid, file_name):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, .9, .9], projection='3d')
    ax.scatter(fix_pts[:, 0],  fix_pts[:, 1], fix_pts[:, 2], color='C1', label='Fixed')
    ax.scatter(mov_pts[:, 0],  mov_pts[:, 1], mov_pts[:, 2], color='C2', label='Moving')
    ax.scatter(fix_centroid[0],  fix_centroid[1], fix_centroid[2], color='none', s=100, edgecolor='b', label='Centroid')
    ax.scatter(mov_centroid[0],  mov_centroid[1], mov_centroid[2], color='none', s=100, edgecolor='b')
    ax.scatter(fix_centroid[0],  fix_centroid[1], fix_centroid[2], color='C1')
    ax.scatter(mov_centroid[0],  mov_centroid[1], mov_centroid[2], color='C2')

    x_range = [np.min(np.hstack([fix_pts[:, 0], mov_pts[:, 0], fix_centroid[0], mov_centroid[0]])),
               np.max(np.hstack([fix_pts[:, 0], mov_pts[:, 0], fix_centroid[0], mov_centroid[0]]))]
    y_range = [np.min(np.hstack([fix_pts[:, 1], mov_pts[:, 1], fix_centroid[1], mov_centroid[1]])),
               np.max(np.hstack([fix_pts[:, 1], mov_pts[:, 1], fix_centroid[1], mov_centroid[1]]))]
    z_range = [np.min(np.hstack([fix_pts[:, 2], mov_pts[:, 2], fix_centroid[2], mov_centroid[2]])),
               np.max(np.hstack([fix_pts[:, 2], mov_pts[:, 2], fix_centroid[2], mov_centroid[2]]))]
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])

    remove_tick_labels(ax, True)
    add_axes_arrows_3d(ax, x_color='r', y_color='g', z_color='b', arrow_length=25, dist_arrow_text=3)
    ax.view_init(None, 45)
    ax.legend(fontsize='xx-large')
    fig.savefig(file_name + '.png')
    fig.savefig(file_name + '.pdf')
    plt.close()

