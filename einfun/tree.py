"""Binary tree data structure.
"""

from collections import deque


def _root_if_none(func):
    """Decorator to return the root if no node is given."""

    def wrapper(self, node=None, *args, **kwargs):
        if node is None:
            node = self.root
        return func(self, node, *args, **kwargs)

    return wrapper


class Node:
    """Node in a tree."""

    def __init__(self, data, name=None, children=None, parent=None):
        """Initialise the object."""
        self.data = data
        self.name = name
        self.children = tuple(children) if children is not None else ()
        self.parent = parent

    def __eq__(self, other):
        """Check if two nodes are equal."""
        return self.data == other.data and self.children == other.children

    def __repr__(self):
        """Return a string representation of the node."""
        if self.name is not None:
            return self.name
        elif isinstance(self.data, str):
            return self.data
        else:
            return repr(self.data)


class Tree:
    """Tree data structure."""

    def __init__(self):
        """Initialise the object."""
        self.nodes = {}

    def add(self, data, name=None, children=None):
        """Add a node to the tree."""

        # Initialise the node if needed
        if data not in self.nodes:
            self.nodes[data] = Node(data, name=name)
        if name is not None:
            self.nodes[data].name = name

        # Add the children
        if children is not None:
            for child in children:
                if child not in self.nodes:
                    self.nodes[child] = Node(child, parent=self.nodes[data])
                self.nodes[data].children += (self.nodes[child],)

    def remove(self, data):
        """Remove a node from the tree."""

        # Remove the node
        del self.nodes[data]

        # Update nodes pointing to the node
        for node in self.nodes:
            if data in self.nodes[node].children:
                self.nodes[node].children.remove(data)

    @_root_if_none
    def height(self, node=None):
        """Calculate the height of a node."""
        if node.parent is None:
            return 0
        return 1 + self.height(node.parent)

    @_root_if_none
    def bfs(self, node=None, return_depth=False):
        """Breadth-first search of the tree."""
        queue = deque([(node, 0)])
        while queue:
            node, depth = queue.popleft()
            if return_depth:
                yield node, depth
            else:
                yield node
            for child in node.children:
                queue.append((child, depth + 1))

    @_root_if_none
    def dfs(self, node=None, return_depth=False):
        """Depth-first search of the tree."""
        stack = [(node, depth)]
        while stack:
            node = stack.pop()
            if return_depth:
                yield node, depth
            else:
                yield node
            for child in node.children:
                stack.append((child, depth + 1))

    @_root_if_none
    def preorder(self, node=None):
        """Preorder traversal of the tree."""
        yield node
        for child in node.children:
            yield from self.preorder(child)

    @_root_if_none
    def postorder(self, node=None):
        """Postorder traversal of the tree."""
        for child in node.children:
            yield from self.postorder(child)
        yield node

    @_root_if_none
    def inorder(self, node=None):
        """Inorder traversal of the tree."""
        for child in node.children[: (len(node.children) + 1) // 2]:
            yield from self.inorder(child)
        yield node
        for child in node.children[(len(node.children) + 1) // 2 :]:
            yield from self.inorder(child)

    @property
    def root(self):
        """Return the root of the tree."""
        for node in self.nodes:
            if self.nodes[node].parent is None:
                return self.nodes[node]

    def __len__(self):
        """Calculate the number of nodes in the tree."""
        return len(self.nodes)

    def __eq__(self, other):
        """Check if two trees are equal."""
        return self.nodes == other.nodes

    def __in__(self, node_or_tree):
        """Check if a node is in the tree, or if a tree is a subtree.
        """
        if isinstance(node_or_tree, Tree):
            node_or_tree = node_or_tree.root
        return node_or_tree in self.nodes

    def __repr__(self):
        """Return a string representation of the tree."""

        # Get the heights for the tree
        heights = {node: self.height(self.nodes[node]) for node in self.nodes}
        max_height = max(heights.values())

        # Get the strings for each tree layer in a nested list
        lines = [[] for _ in range(max_height + 1)]
        for node in self.inorder(self.root):
            level = heights[node.data]
            lines[level].append(node)
            for i in range(len(lines)):
                if i != level:
                    lines[i].append(" " * (len(repr(node)) + 1))

        # Find the sizes of each cell
        def node_repr(node):
            """Node representation with a space to the left."""
            if node.name is not None:
                return " " + node.name
            elif isinstance(node.data, str):
                return " " + node.data
            else:
                return " " + repr(node.data)

        sizes = [len(node_repr(node) if isinstance(node, Node) else node) for node in lines[-1]]
        sizes_l = [size // 2 for size in sizes]
        sizes_r = [size - size // 2 - 1 for size in sizes]

        # Determine the separators
        seps = []
        for above, below in zip(lines[:-1], lines[1:]):
            # Record the separator for this level
            sep = [" " * size for size in sizes]

            # Loop through cells in the level above
            for i, node in enumerate(above):
                # If it's not a node or it has no children, skip it
                if not isinstance(node, Node) or len(node.children) == 0:
                    continue

                # Find the children cells of the node
                children = []
                for j in range(len(above)):
                    if isinstance(below[j], Node):
                        if below[j] in node.children:
                            children.append(j)

                # Write the connection to the leftmost child
                idx = children[0]
                sep[idx] = " " * sizes_l[idx] + "┌" + "─" * sizes_r[idx]

                # Write the connections to the middle children
                for j in range(children[0] + 1, children[-1]):
                    if i == j:
                        sep[j] = "─" * sizes_l[j] + "┴" + "─" * sizes_r[j]
                    elif j in children:
                        sep[j] = "─" * sizes_l[j] + "┬" + "─" * sizes_r[j]
                    else:
                        sep[j] = "─" * sizes[j]

                # Write the connection to the rightmost child
                if children[0] != children[-1]:
                    idx = children[-1]
                    sep[idx] = "─" * sizes_l[idx] + "┐" + " " * sizes_r[idx]
                else:
                    idx = children[-1] + 1
                    sep[idx] = "─" * sizes_l[idx] + "┘" + " " * sizes_r[idx]

            seps.append(sep)

        # Combine the lines and separators
        combined = []
        for i in range(len(seps)):
            combined.append(lines[i])
            combined.append(seps[i])
        combined.append(lines[-1])

        # Convert the nodes to strings
        for i, line in enumerate(combined):
            for j, node in enumerate(line):
                if isinstance(node, Node):
                    combined[i][j] = node_repr(node)

        # Join the lines
        out = ""
        for i in range(len(seps)):
            out += "".join([str(x) for x in lines[i]]) + "\n"
            out += "".join([str(x) for x in seps[i]]) + "\n"
        out += "".join([str(x) for x in lines[-1]])

        return out
