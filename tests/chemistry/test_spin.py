import pytest

from albert._core import Expression, Mul, Tensor
from albert.chemistry import expand_antisymmetrized_eris, spin_integrate
from albert.chemistry.factory import TensorFactory
from albert.chemistry.metadata import Metadata
from albert.chemistry.options import ReferenceType


@pytest.fixture
def metadata() -> Metadata:
    return Metadata()


@pytest.fixture
def spin_metadata(metadata: Metadata) -> Metadata:
    return metadata.with_spin()


def _tensor(symbol: str, *indices: str) -> Tensor:
    return TensorFactory(symbol)[indices]


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


def test_uhf_to_rhf_removes_spin_suffixes(spin_metadata: Metadata) -> None:
    i = spin_metadata.indices["o_α"][0]
    a = spin_metadata.indices["v_α"][0]
    expression = Expression(_tensor("t1", i, a), _tensor("t1", i, a))

    result = spin_integrate(expression, ReferenceType.UHF, ReferenceType.RHF)

    assert len(result) == 1
    assert result[0].lhs.indices == [spin_metadata.indices["o"][0], spin_metadata.indices["v"][0]]
    assert result[0].rhs == result[0].lhs


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
