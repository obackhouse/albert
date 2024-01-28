"""Miscellaneous functions.
"""

import networkx as nx


def plot_graph(graph):
    """Plot the graph representing the contractions in an expression.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The graph to plot.
    """

    # Import matplotlib here to avoid dependency
    import matplotlib.pyplot as plt

    # Initialise the figure
    fig, ax = plt.subplots()

    # Get the layout
    pos = nx.kamada_kawai_layout(graph)

    # Draw the nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="w",
        edgecolors="k",
        linewidths=1,
        node_size=500,
    )

    # Draw the edges
    for edge in graph.edges:
        print(graph.nodes[edge[0]], graph.nodes[edge[1]], graph.edges[edge])
        i, j = divmod(edge[2], 2)
        ax.annotate(
            "",
            xy=pos[edge[0]],
            xycoords="data",
            xytext=pos[edge[1]],
            textcoords="data",
            zorder=-1,
            arrowprops=dict(
                arrowstyle="-",
                color="0.5",
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                connectionstyle=f"arc3,rad={0.05 * (i + 1) * (-1)**j}",
            ),
        )

    # Draw the labels
    nx.draw_networkx_labels(graph, pos, ax=ax, labels={node: node[0] for node in graph.nodes})

    # Show the figure
    plt.show()
