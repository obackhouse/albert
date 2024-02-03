"""Miscellaneous functions.
"""

import networkx as nx


class TwoWayDict(dict):
    """A dictionary that can be accessed by key or value.

    Parameters
    ----------
    *args : tuple
        The key-value pairs to initialise the dictionary with.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the object."""
        super().__init__(*args, **kwargs)
        self.update({v: k for k, v in self.items()})

    def __setitem__(self, key, value):
        """Set a value and its key."""
        super().__setitem__(key, value)
        super().__setitem__(value, key)

    def __delitem__(self, key):
        """Delete a value and its key."""
        super().__delitem__(self[key])
        super().__delitem__(key)

    def __len__(self):
        """Return the length of the dictionary."""
        return super().__len__() // 2


def plot_graph(graph, ax=None, show=True):
    """Plot the graph representing the contractions in an expression.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The graph to plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, a new figure is created.
        Default value is `None`.
    show : bool, optional
        Whether to show the figure. Default value is `True`.
    """

    # Import matplotlib here to avoid dependency
    import matplotlib.pyplot as plt

    # Initialise the figure
    if ax is None:
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
    for n, edge in enumerate(graph.edges):
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
                color=f"C{n}",
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                connectionstyle=f"arc3,rad={0.15 * (i + 1) * (-1)**j}",
            ),
        )
        label = f"[{graph.nodes[edge[0]]['data'].name}, {graph.edges[edge]['data'][edge[0]]}]"
        label += r"$ \rightarrow $"
        label += f"[{graph.nodes[edge[1]]['data'].name}, {graph.edges[edge]['data'][edge[1]]}]"
        ax.plot(
            [pos[edge[0]][0], pos[edge[0]][0]],
            [pos[edge[0]][1], pos[edge[0]][1]],
            color=f"C{n}",
            label=label,
            zorder=-1,
        )

    # Draw the labels
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels={node: graph._node[node]["data"].name for node in graph.nodes},
    )

    # Show the figure
    if show:
        plt.axis("off")
        plt.legend()
        plt.show()

    return ax
