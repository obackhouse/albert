"""Miscellaneous functions.
"""

from collections import deque

import networkx as nx


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


class BinaryTreeNode:
    """Node in a binary tree."""

    def __init__(self, data, name=None, left=None, right=None, parent=None):
        """Initialise a binary tree node."""
        self.data = data
        self.name = name
        self.left = left
        self.right = right
        self.parent = None

    @property
    def children(self):
        """Return the children of the binary tree node."""
        return [child for child in (self.left, self.right) if child]

    def __eq__(self, other):
        """Check if two binary tree nodes are equal."""
        return self.data == other.data and self.left == other.left and self.right == other.right

    def __repr__(self):
        """Return a string representation of the binary tree node."""
        return self.name if self.name is not None else repr(self.data)

    def _display_aux(self):
        """Auxilary function for displaying the binary tree."""

        # Characters
        vertl = "│"
        vertr = "│"
        line = "─"

        # Childless
        if self.right is None and self.left is None:
            line = repr(self)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = repr(self)
            u = len(s)
            line1 = f"{(x + 1) * ' '}{(n - x - 1) * line}┐{s}"
            line2 = f"{x * ' '}{vertl}{(n - x - 1 + u) * ' '}"
            shifted_lines = [line + u * " " for line in lines]
            return [line1, line2] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = repr(self)
            u = len(s)
            line1 = f"{x * ' '}┌{vertl}{(n - x - 1 + u) * ' '}"
            line2 = f"{x * ' '}{(n - x - 1) * line}{s}"
            shifted_lines = [u * " " + line for line in lines]
            return [line1, line2] + shifted_lines, n + u, p + 2, u // 2

        # Two children
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = repr(self)
        u = len(s)
        line1 = f"{x * ' '}┌{(n - x - 1) * line}{s}{y * line}┐{(m - y - 1) * ' '}"
        line2 = f"{x * ' '}{vertl}{(n - x - 1 + u + y) * ' '}{vertr}{(m - y - 1) * ' '}"
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = [line1, line2] + [a + u * " " + b for a, b in zipped_lines]

        return lines, n + m + u, max(p, q) + 2, n + u // 2


class BinaryTree:
    """Binary tree."""

    def __init__(self):
        """Initialise a binary tree."""
        self.nodes = {}
        self.contractions = {}

    def add(
        self,
        data,
        name=None,
        left=None,
        right=None,
        parent=None,
    ):
        """Add a node to the binary tree."""

        # Add the node
        if data not in self.nodes:
            self.nodes[data] = BinaryTreeNode(data, left=left, right=right, parent=parent)
        if name is not None:
            self.nodes[data].name = name

        # Add the left child
        if left is None:
            self.nodes[data].left = None
        else:
            if left not in self.nodes:
                self.nodes[left] = BinaryTreeNode(left, parent=data)
            self.nodes[data].left = self.nodes[left]

        # Add the right child
        if right is None:
            self.nodes[data].right = None
        else:
            if right not in self.nodes:
                self.nodes[right] = BinaryTreeNode(right, parent=data)
            self.nodes[data].right = self.nodes[right]

    def remove(self, data):
        """Remove a node from the binary tree."""

        # Remove the node
        del self.nodes[data]

        # Update nodes pointing to the node
        for node in self.nodes:
            if self.nodes[node].left == data:
                del self.nodes[node].left
            if self.nodes[node].right == data:
                del self.nodes[node].right

    def height(self, node):
        """Calculate the height of a node."""
        if not (node and node.children):
            return 0
        return 1 + max(self.height(child) for child in node.children)

    def bfs(self, node):
        """Breadth-first search of the binary tree."""
        queue = deque([node])
        while queue:
            node = queue.popleft()
            yield node
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

    def dfs(self, node):
        """Depth-first search of the binary tree."""
        stack = [node]
        while stack:
            node = stack.pop()
            yield node
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

    def preorder(self, node):
        """Preorder traversal of the binary tree."""
        yield node
        if node.left is not None:
            yield from self.preorder(node.left)
        if node.right is not None:
            yield from self.preorder(node.right)

    def inorder(self, node):
        """Inorder traversal of the binary tree."""
        if node.left is not None:
            yield from self.inorder(node.left)
        yield node
        if node.right is not None:
            yield from self.inorder(node.right)

    def postorder(self, node):
        """Postorder traversal of the binary tree."""
        if node.left is not None:
            yield from self.postorder(node.left)
        if node.right is not None:
            yield from self.postorder(node.right)
        yield node

    def __len__(self):
        """Calculate the number of nodes in the binary tree."""
        return len(self.nodes)

    def __eq__(self, other):
        """Check if two binary trees are equal."""
        return self.nodes == other.nodes

    def __repr__(self):
        """Return a string representation of the binary tree."""
        node = next(iter(self.nodes))
        while self.nodes[node].parent is not None:
            node = self.nodes[node].parent
        lines, *_ = self.nodes[node]._display_aux()
        return "\n".join(lines)
