from albert import Add, Mul, Node, NodeKind, Scalar, SearchOrder, Tensor


def _expression() -> Add:
    return Add([Tensor("A", ["i"]), Mul([Tensor("B", ["i", "j"]), Tensor("C", ["j"]), Scalar(2.0)])])


def test_search_visits_nodes_in_preorder_and_postorder() -> None:
    expression = _expression()

    def is_node(_: Node) -> bool:
        return True

    preorder = expression.search(is_node)
    postorder = expression.search(is_node, order=SearchOrder.POSTORDER)

    assert [node.kind for node in preorder] == [
        NodeKind.ADD,
        NodeKind.TENSOR,
        NodeKind.MUL,
        NodeKind.TENSOR,
        NodeKind.TENSOR,
        NodeKind.SCALAR,
    ]
    assert [node.kind for node in postorder] == [
        NodeKind.TENSOR,
        NodeKind.TENSOR,
        NodeKind.TENSOR,
        NodeKind.SCALAR,
        NodeKind.MUL,
        NodeKind.ADD,
    ]


def test_search_and_find_obey_max_depth() -> None:
    expression = _expression()

    def is_tensor(node: Node) -> bool:
        return isinstance(node, Tensor)

    assert expression.search(is_tensor, max_depth=0) == []
    assert len(expression.search(is_tensor, max_depth=1)) == 1
    assert isinstance(expression.find(is_tensor, max_depth=1), Tensor)
    assert expression.find(is_tensor, max_depth=0) is None


def test_find_stops_at_first_match() -> None:
    expression = _expression()
    visited: list[NodeKind] = []

    def predicate(node) -> bool:
        visited.append(node.kind)
        return isinstance(node, Tensor)

    result = expression.find(predicate)

    assert isinstance(result, Tensor)
    assert result.symbol == "A"
    assert visited == [NodeKind.ADD, NodeKind.TENSOR]


def test_delete_replaces_matches_with_zero_scalars() -> None:
    expression = _expression()

    result = expression.delete(lambda node: isinstance(node, Tensor))

    assert isinstance(result, Add)
    assert all(isinstance(child, Scalar) and child.value == 0.0 for child in result.children[0:1])
    assert isinstance(result.children[1], Mul)
    assert all(isinstance(child, Scalar) for child in result.children[1].children)


def test_delete_respects_max_depth() -> None:
    expression = _expression()

    result = expression.delete(lambda node: isinstance(node, Tensor), max_depth=0)

    assert result is expression


def test_apply_identity_reuses_original_nodes() -> None:
    expression = _expression()

    result = expression.apply(lambda node: node)

    assert result is expression
    assert result.children[1] is expression.children[1]


def test_apply_rebuilds_only_changed_path() -> None:
    expression = _expression()
    original_first = expression.children[0]
    original_second = expression.children[1]

    result = expression.apply(
        lambda node: Tensor("X", node.indices) if isinstance(node, Tensor) and node.symbol == "A" else node,
        predicate=lambda node: isinstance(node, Tensor) and node.symbol == "A",
    )

    assert isinstance(result, Add)
    assert result is not expression
    assert result.children[0].symbol == "X"
    assert result.children[1] is original_second
    assert original_first.symbol == "A"


def test_apply_respects_max_depth() -> None:
    expression = _expression()

    result = expression.apply(
        lambda node: Tensor("X", node.indices) if isinstance(node, Tensor) else node,
        max_depth=1,
    )
    assert result is not expression
    assert result.children[0].symbol == "X"
    assert result.children[1] is expression.children[1]
