from itertools import product

import pytest

from albert._core import Add, Expression, Mul, Node, Tensor, simplify
from albert.chemistry import expand_antisymmetrized_eris, spin_integrate
from albert.chemistry.factory import TensorFactory
from albert.chemistry.metadata import Metadata
from albert.chemistry.options import OrbitalType, ReferenceType
from albert.chemistry.spin import _collect_scalar_dummy_permutations


@pytest.fixture
def metadata() -> Metadata:
    return Metadata()


@pytest.fixture
def spin_metadata(metadata: Metadata) -> Metadata:
    return metadata.with_spin()


def _tensor(symbol: str, *indices: str) -> Tensor:
    return TensorFactory(symbol)[indices]


def _spin_index(index: str, spin: OrbitalType) -> str:
    """Return a default chemistry index with the requested spin label."""
    return index + spin.subscript


def _terms(node: Node) -> set[str]:
    """Return formatted additive terms without relying on their order."""
    simplified = simplify(node)
    children = simplified.children if isinstance(simplified, Add) else [simplified]
    return {term.format() for term in children}


def test_ghf_to_uhf_maps_one_body_tensor_into_spin_blocks(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    expression = Expression(_tensor("h", i, j), _tensor("h", i, j))

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.UHF)

    alpha_i, alpha_j = spin_metadata.indices["o_α"][:2]
    beta_i, beta_j = spin_metadata.indices["o_β"][:2]
    assert [equation.lhs.indices for equation in result] == [
        [alpha_i, alpha_j],
        [beta_i, beta_j],
    ]
    assert all(equation.rhs == equation.lhs for equation in result)


def test_ghf_to_uhf_prunes_inconsistent_repeated_indices(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    a = spin_metadata.indices["v_γ"][0]
    expression = Expression(
        _tensor("t1", i, a),
        Mul([_tensor("h", i, j), _tensor("f", j, a)]),
    )

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.UHF)

    alpha_i, alpha_j = spin_metadata.indices["o_α"][:2]
    alpha_a = spin_metadata.indices["v_α"][0]
    beta_i, beta_j = spin_metadata.indices["o_β"][:2]
    beta_a = spin_metadata.indices["v_β"][0]
    assert len(result) == 2
    assert {tuple(tuple(term.indices) for term in equation.rhs.children) for equation in result} == {
        (
            tuple(_tensor("h", alpha_i, alpha_j).indices),
            tuple(_tensor("f", alpha_j, alpha_a).indices),
        ),
        (
            tuple(_tensor("h", beta_i, beta_j).indices),
            tuple(_tensor("f", beta_j, beta_a).indices),
        ),
    }


def test_ghf_to_uhf_uses_physicists_eri_spin_pairs(spin_metadata: Metadata) -> None:
    occupied = spin_metadata.indices["o_γ"][:2]
    virtual = spin_metadata.indices["v_γ"][:2]
    indices = [occupied[0], virtual[0], occupied[1], virtual[1]]
    expression = Expression(
        _tensor("t2", *indices),
        _tensor("v", *indices),
    )

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.UHF)

    o_alpha, o_beta = spin_metadata.indices["o_α"], spin_metadata.indices["o_β"]
    v_alpha, v_beta = spin_metadata.indices["v_α"], spin_metadata.indices["v_β"]
    assert {tuple(equation.rhs.indices) for equation in result} == {
        (o_alpha[0], v_alpha[0], o_alpha[1], v_alpha[1]),
        (o_alpha[0], v_beta[0], o_alpha[1], v_beta[1]),
        (o_beta[0], v_alpha[0], o_beta[1], v_alpha[1]),
        (o_beta[0], v_beta[0], o_beta[1], v_beta[1]),
    }


def test_ghf_to_rhf_combines_spin_blocks(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    expression = Expression(_tensor("h", i, j), _tensor("h", i, j))

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.RHF)

    assert len(result) == 1
    spatial_i, spatial_j = spin_metadata.indices["o"][:2]
    assert result[0].lhs.indices == [spatial_i, spatial_j]
    assert result[0].rhs == 2 * _tensor("h", spatial_i, spatial_j)


def test_ghf_to_rhf_preserves_lhs_index_order(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    expression = Expression(_tensor("h", i, j), _tensor("h", j, i))

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.RHF)

    spatial_i, spatial_j = spin_metadata.indices["o"][:2]
    assert result[0].lhs.indices == [spatial_i, spatial_j]
    assert result[0].rhs == 2 * _tensor("h", spatial_j, spatial_i)


def test_spin_integrate_uses_transform_like_for_unknown_lhs(spin_metadata: Metadata) -> None:
    i = spin_metadata.indices["o_γ"][0]
    a = spin_metadata.indices["v_γ"][0]
    expression = Expression(Tensor("t1new", [i, a]), _tensor("f", i, a))

    with pytest.raises(NotImplementedError, match="t1new"):
        spin_integrate(expression, ReferenceType.GHF, ReferenceType.RHF)

    result = spin_integrate(
        expression,
        ReferenceType.GHF,
        ReferenceType.RHF,
        transform_like={"t1new": "t1"},
    )

    spatial_i = spin_metadata.indices["o"][0]
    spatial_a = spin_metadata.indices["v"][0]
    assert len(result) == 1
    assert result[0].lhs == Tensor("t1new", [spatial_i, spatial_a])
    assert result[0].rhs == 2 * _tensor("f", spatial_i, spatial_a)


def test_ghf_to_uhf_projects_ccsd_t1_onto_alpha_alpha(spin_metadata: Metadata) -> None:
    i = spin_metadata.indices["o_γ"][0]
    a = spin_metadata.indices["v_γ"][0]
    expression = Expression(Tensor("t1new", [i, a]), _tensor("f", i, a))

    result = spin_integrate(
        expression,
        ReferenceType.GHF,
        ReferenceType.UHF,
        transform_like={"t1new": "t1"},
        index_spins={"i": "a", "a": "a"},
    )

    assert len(result) == 1
    assert result[0].lhs.indices == [spin_metadata.indices["o_α"][0], spin_metadata.indices["v_α"][0]]
    assert result[0].rhs == _tensor("f", spin_metadata.indices["o_α"][0], spin_metadata.indices["v_α"][0])


def test_ghf_to_uhf_projects_ccsd_t2_onto_alpha_beta_alpha_beta(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    a, b = spin_metadata.indices["v_γ"][:2]
    indices = [i, j, a, b]
    expression = Expression(Tensor("t2new", indices), Tensor("v", indices))

    result = spin_integrate(
        expression,
        ReferenceType.GHF,
        ReferenceType.UHF,
        transform_like={"t2new": "t2"},
        index_spins={"i": "a", "j": "b", "a": "a", "b": "b"},
    )

    expected = [
        spin_metadata.indices["o_α"][0],
        spin_metadata.indices["o_β"][1],
        spin_metadata.indices["v_α"][0],
        spin_metadata.indices["v_β"][1],
    ]
    assert len(result) == 1
    assert result[0].lhs.indices == expected
    assert result[0].rhs.indices == expected


def test_spin_integrate_uses_transform_like_for_rhs_tensors(spin_metadata: Metadata) -> None:
    i = spin_metadata.indices["o_γ"][0]
    a = spin_metadata.indices["v_γ"][0]
    expression = Expression(_tensor("t1", i, a), Tensor("t1new", [i, a]))

    result = spin_integrate(
        expression,
        ReferenceType.GHF,
        ReferenceType.UHF,
        transform_like={"t1new": "t1"},
    )

    assert len(result) == 2
    assert all(equation.rhs.symbol == "t1new" for equation in result)


def test_spin_integrate_uses_transform_like_for_doubles_spin_adaptation(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_α"][:2]
    a, b = spin_metadata.indices["v_α"][:2]
    expression = Expression(Tensor("t2new", [i, j, a, b]), Tensor("t2new", [i, j, a, b]))

    result = spin_integrate(
        expression,
        ReferenceType.UHF,
        ReferenceType.RHF,
        transform_like={"t2new": "t2"},
    )

    spatial_i, spatial_j = spin_metadata.indices["o"][:2]
    spatial_a, spatial_b = spin_metadata.indices["v"][:2]
    assert result[0].lhs == Tensor("t2new", [spatial_i, spatial_j, spatial_a, spatial_b])
    assert result[0].rhs == Tensor("t2new", [spatial_i, spatial_j, spatial_a, spatial_b]) - Tensor(
        "t2new", [spatial_i, spatial_j, spatial_b, spatial_a]
    )


def test_spin_integrate_expands_transform_like_antisymmetrized_eris(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    a, b = spin_metadata.indices["v_γ"][:2]
    indices = [i, j, a, b]
    expression = Expression(_tensor("t2", *indices), Tensor("v_custom_a", indices))

    result = spin_integrate(
        expression,
        ReferenceType.GHF,
        ReferenceType.UHF,
        transform_like={"v_custom_a": "v_a"},
    )

    assert result
    assert all(
        {tensor.symbol for tensor in equation.rhs.search(lambda node: isinstance(node, Tensor))} == {"v"}
        for equation in result
    )


def test_spin_integrate_validates_transform_like_rank(spin_metadata: Metadata) -> None:
    i = spin_metadata.indices["o_γ"][0]
    a = spin_metadata.indices["v_γ"][0]
    expression = Expression(Tensor("t1new", [i, a]), _tensor("f", i, a))

    with pytest.raises(ValueError, match="has rank 2.*reference 't2' has rank 4"):
        spin_integrate(
            expression,
            ReferenceType.GHF,
            ReferenceType.UHF,
            transform_like={"t1new": "t2"},
        )


def test_uhf_to_rhf_removes_spin_suffixes(spin_metadata: Metadata) -> None:
    i = spin_metadata.indices["o_α"][0]
    a = spin_metadata.indices["v_α"][0]
    expression = Expression(_tensor("t1", i, a), _tensor("t1", i, a))

    result = spin_integrate(expression, ReferenceType.UHF, ReferenceType.RHF)

    assert len(result) == 1
    assert result[0].lhs.indices == [spin_metadata.indices["o"][0], spin_metadata.indices["v"][0]]
    assert result[0].rhs == result[0].lhs


def test_uhf_to_rhf_preserves_lhs_index_order(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_α"][:2]
    expression = Expression(_tensor("h", i, j), _tensor("h", j, i))

    result = spin_integrate(expression, ReferenceType.UHF, ReferenceType.RHF)

    spatial_i, spatial_j = spin_metadata.indices["o"][:2]
    assert result[0].lhs.indices == [spatial_i, spatial_j]
    assert result[0].rhs == _tensor("h", spatial_j, spatial_i)


def test_antisymmetrized_eri_filter_expands_to_direct_and_exchange_terms(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    a, b = spin_metadata.indices["v_γ"][:2]
    tensor = _tensor("v_a", i, j, a, b)

    expanded = tensor.apply(expand_antisymmetrized_eris)

    assert expanded.format() == f"v[{i}, {j}, {a}, {b}] - v[{i}, {j}, {b}, {a}]"


def test_spin_integration_expands_antisymmetrized_eris_first(spin_metadata: Metadata) -> None:
    i, j = spin_metadata.indices["o_γ"][:2]
    a, b = spin_metadata.indices["v_γ"][:2]
    expression = Expression(
        _tensor("t2", i, j, a, b),
        _tensor("v_a", i, j, a, b),
    )

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.UHF)

    assert result
    assert all(
        not equation.rhs.search(lambda node: isinstance(node, Tensor) and node.symbol == "v_a") for equation in result
    )


def test_spin_integrates_ccd_doubles_residual_fragment(spin_metadata: Metadata) -> None:
    occupied = spin_metadata.indices["o_γ"][:4]
    virtual = spin_metadata.indices["v_γ"][:4]
    i, j, k, ell = occupied
    a, b, c, d = virtual
    t2 = TensorFactory("t2")
    v_a = TensorFactory("v_a")
    expression = Expression(
        t2[a, b, i, j],
        Add(
            [
                v_a[a, b, i, j],
                0.5 * Mul([v_a[k, ell, i, j], t2[a, b, k, ell]]),
                0.5 * Mul([v_a[a, b, d, c], t2[d, c, i, j]]),
            ]
        ),
    )

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.UHF)

    assert len(result) == 4
    assert {tuple(index[-2:] for index in equation.lhs.indices) for equation in result} == {
        ("_α", "_α", "_α", "_α"),
        ("_α", "_β", "_α", "_β"),
        ("_β", "_α", "_β", "_α"),
        ("_β", "_β", "_β", "_β"),
    }
    assert all(equation.lhs.indices == equation.rhs.external_indices for equation in result)


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (ReferenceType.RHF, ReferenceType.UHF),
        (ReferenceType.UHF, ReferenceType.GHF),
        (ReferenceType.RHF, ReferenceType.GHF),
    ],
)
def test_spin_integrate_rejects_upward_transforms(src: ReferenceType, dst: ReferenceType, metadata: Metadata) -> None:
    i, j = metadata.indices["o"][:2]
    expression = Expression(_tensor("h", i, j), _tensor("h", i, j))

    with pytest.raises(ValueError, match="unsupported spin integration"):
        spin_integrate(expression, src, dst)


def test_mp2_energy_spin_integrates_to_known_uhf_expression() -> None:
    metadata = Metadata()
    occupied = metadata.indices["o"][:2]
    virtual = metadata.indices["v"][:2]
    i, j = occupied
    a, b = virtual
    v_a = TensorFactory("v_a")
    t2_a = TensorFactory("t2_a")
    energy = Expression(
        TensorFactory("r0")[()],
        0.25 * Mul([v_a[i, j, a, b], t2_a[i, j, a, b]]),
    )

    result = spin_integrate(energy, ReferenceType.GHF, ReferenceType.UHF)

    expected_terms = []
    spins = (OrbitalType.ALPHA, OrbitalType.BETA)
    for spin_i, spin_j, spin_a, spin_b in product(spins, repeat=4):
        if sorted((spin_i, spin_j)) != sorted((spin_a, spin_b)):
            continue
        indices = tuple(
            _spin_index(index, spin) for index, spin in zip((i, j, a, b), (spin_i, spin_j, spin_a, spin_b), strict=True)
        )
        if spin_i is spin_a and spin_j is spin_b:
            expected_terms.append(0.25 * Mul([TensorFactory("v")[indices], t2_a[indices]]))
        if spin_i is spin_b and spin_j is spin_a:
            exchange_indices = (indices[0], indices[1], indices[3], indices[2])
            expected_terms.append(-0.25 * Mul([TensorFactory("v")[exchange_indices], t2_a[indices]]))

    assert len(result) == 1
    assert result[0].lhs.symbol == "r0"
    assert _terms(result[0].rhs) == _terms(Add(expected_terms))


def test_ghf_mp2_energy_spin_integrates_to_closed_shell_rhf_expression() -> None:
    metadata = Metadata()
    i, j = metadata.indices["o"][:2]
    a, b = metadata.indices["v"][:2]
    expression = Expression(
        Tensor("E", []),
        0.25 * Mul([_tensor("t2", i, j, b, a), _tensor("v_a", i, j, b, a)]),
    )

    result = spin_integrate(expression, ReferenceType.GHF, ReferenceType.RHF, metadata)

    assert len(result) == 1
    assert result[0].format() == (
        f"E = 2 t2[{i}, {j}, {b}, {a}] v[{i}, {j}, {b}, {a}] - t2[{i}, {j}, {b}, {a}] v[{i}, {j}, {a}, {b}]"
    )


def test_scalar_dummy_collection_does_not_rename_between_spaces() -> None:
    metadata = Metadata(indices={"o": ["i"], "v": ["a"]}, sizes={"o": 2, "v": 3})
    expression = Add(
        [
            Mul([Tensor("A", ["i"]), Tensor("B", ["i"])]),
            Mul([Tensor("A", ["a"]), Tensor("B", ["a"])]),
        ]
    )

    result = _collect_scalar_dummy_permutations(expression, metadata)

    assert _terms(result) == {"A[i] B[i]", "A[a] B[a]"}


def test_mp2_occupied_1rdm_spin_integrates_to_known_uhf_expression() -> None:
    metadata = Metadata()
    occupied = metadata.indices["o"][:3]
    virtual = metadata.indices["v"][:2]
    i, j, k = occupied
    a, b = virtual
    d1 = TensorFactory("d1")
    l2_a = TensorFactory("l2_a")
    t2_a = TensorFactory("t2_a")
    density = Expression(
        d1[i, j],
        -0.5 * Mul([l2_a[i, k, a, b], t2_a[j, k, a, b]]),
    )

    result = spin_integrate(density, ReferenceType.GHF, ReferenceType.UHF)

    expected: dict[tuple[str, ...], list[Node]] = {}
    for spin_i in (OrbitalType.ALPHA, OrbitalType.BETA):
        i_spin = _spin_index(i, spin_i)
        j_spin = _spin_index(j, spin_i)
        for spin_k, spin_a, spin_b in product((OrbitalType.ALPHA, OrbitalType.BETA), repeat=3):
            if sorted((spin_i.subscript, spin_k.subscript)) != sorted((spin_a.subscript, spin_b.subscript)):
                continue
            k_spin = _spin_index(k, spin_k)
            a_spin = _spin_index(a, spin_a)
            b_spin = _spin_index(b, spin_b)
            expected.setdefault((i_spin, j_spin), []).append(
                -0.5 * Mul([l2_a[i_spin, k_spin, a_spin, b_spin], t2_a[j_spin, k_spin, a_spin, b_spin]])
            )

    assert len(result) == 2
    assert {tuple(expression.lhs.indices) for expression in result} == {
        (f"{i}_α", f"{j}_α"),
        (f"{i}_β", f"{j}_β"),
    }
    for expression in result:
        assert _terms(expression.rhs) == _terms(Add(expected[tuple(expression.lhs.indices)]))


def test_mp2_virtual_1rdm_spin_integrates_to_known_uhf_expression() -> None:
    metadata = Metadata()
    occupied = metadata.indices["o"][:2]
    virtual = metadata.indices["v"][:3]
    i, j = occupied
    a, b, c = virtual
    d1 = TensorFactory("d1")
    l2_a = TensorFactory("l2_a")
    t2_a = TensorFactory("t2_a")
    density = Expression(
        d1[a, b],
        0.5 * Mul([l2_a[i, j, a, c], t2_a[i, j, b, c]]),
    )

    result = spin_integrate(density, ReferenceType.GHF, ReferenceType.UHF)

    expected: dict[tuple[str, ...], list[Node]] = {}
    for spin_a in (OrbitalType.ALPHA, OrbitalType.BETA):
        a_spin = _spin_index(a, spin_a)
        b_spin = _spin_index(b, spin_a)
        for spin_i, spin_j, spin_c in product((OrbitalType.ALPHA, OrbitalType.BETA), repeat=3):
            occupied_spins = (spin_i.subscript, spin_j.subscript)
            if sorted(occupied_spins) != sorted((spin_a.subscript, spin_c.subscript)):
                continue
            i_spin = _spin_index(i, spin_i)
            j_spin = _spin_index(j, spin_j)
            c_spin = _spin_index(c, spin_c)
            expected.setdefault((a_spin, b_spin), []).append(
                0.5 * Mul([l2_a[i_spin, j_spin, a_spin, c_spin], t2_a[i_spin, j_spin, b_spin, c_spin]])
            )

    assert len(result) == 2
    assert {tuple(expression.lhs.indices) for expression in result} == {
        (f"{a}_α", f"{b}_α"),
        (f"{a}_β", f"{b}_β"),
    }
    for expression in result:
        assert _terms(expression.rhs) == _terms(Add(expected[tuple(expression.lhs.indices)]))
