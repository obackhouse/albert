#include "evaluate.h"

#include "algebra.h"
#include "sort.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace albert {
namespace {

namespace py = pybind11;
using IndexSpaceMap = std::unordered_map<std::string, std::string>;
using Monomial = std::vector<NodePtr>;

std::size_t checked_product(const std::vector<std::size_t>& shape) {
    std::size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension != 0 && result > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor array size exceeds addressable memory");
        }
        result *= dimension;
    }
    return result;
}

std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> result(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis > 1; --axis) {
        const auto previous = axis - 2;
        if (shape[axis - 1] != 0 &&
            result[axis - 1] > std::numeric_limits<std::size_t>::max() / shape[axis - 1]) {
            throw std::overflow_error("tensor array stride exceeds addressable memory");
        }
        result[previous] = result[axis - 1] * shape[axis - 1];
    }
    return result;
}

IndexSpaceMap make_index_space_map(const IndexSpaces& spaces) {
    IndexSpaceMap result;
    for (const auto& [space, labels] : spaces) {
        for (const auto& label : labels) {
            const auto [entry, inserted] = result.emplace(label, space);
            if (!inserted) {
                throw std::invalid_argument("index " + label +
                                            " belongs to more than one space");
            }
        }
    }
    return result;
}

std::size_t space_size(const std::string& space, const SpaceSizes& sizes) {
    const auto found = sizes.find(space);
    if (found == sizes.end()) {
        throw std::invalid_argument("space " + space + " has no declared size");
    }
    if (found->second > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("space " + space + " exceeds addressable size");
    }
    return static_cast<std::size_t>(found->second);
}

void expand_terms(const NodePtr& node, std::vector<Monomial>& output) {
    output.clear();
    if (node->kind() == NodeKind::Scalar || node->kind() == NodeKind::Tensor) {
        output.push_back({node});
        return;
    }
    if (node->kind() == NodeKind::Add) {
        for (const auto& child : node->children()) {
            std::vector<Monomial> child_terms;
            expand_terms(child, child_terms);
            output.insert(output.end(), std::make_move_iterator(child_terms.begin()),
                          std::make_move_iterator(child_terms.end()));
        }
        return;
    }

    output = {Monomial{}};
    for (const auto& child : node->children()) {
        std::vector<Monomial> child_terms;
        expand_terms(child, child_terms);
        if (child_terms.empty()) {
            output.clear();
            return;
        }
        if (output.size() > std::numeric_limits<std::size_t>::max() / child_terms.size()) {
            throw std::length_error("expanded expression has too many terms to evaluate");
        }
        std::vector<Monomial> products;
        products.reserve(output.size() * child_terms.size());
        for (const auto& left : output) {
            for (const auto& right : child_terms) {
                Monomial product;
                product.reserve(left.size() + right.size());
                product.insert(product.end(), left.begin(), left.end());
                product.insert(product.end(), right.begin(), right.end());
                products.push_back(std::move(product));
            }
        }
        output = std::move(products);
    }
}

template <typename T> struct TensorFactor {
    const DenseTensor<T>* array = nullptr;
    std::vector<std::string> indices;
    std::vector<std::size_t> strides;
};

template <typename T> struct PreparedMonomial {
    T coefficient = T{1};
    std::vector<TensorFactor<T>> tensors;
    std::vector<std::string> external_indices;
    std::vector<std::string> internal_indices;
    std::unordered_map<std::string, std::size_t> dimensions;
};

template <typename T>
PreparedMonomial<T> prepare_monomial(
    const Monomial& monomial, const TensorArrays<T>& arrays,
    const IndexSpaceMap& index_spaces, const SpaceSizes& sizes) {
    PreparedMonomial<T> result;
    std::unordered_map<std::string, std::size_t> counts;
    std::vector<std::string> index_order;
    for (const auto& factor : monomial) {
        if (const auto* scalar = dynamic_cast<const Scalar*>(factor.get())) {
            result.coefficient *= T(scalar->value());
            continue;
        }
        const auto* tensor = dynamic_cast<const Tensor*>(factor.get());
        if (!tensor) {
            throw std::invalid_argument("unsupported node in expanded expression");
        }

        std::vector<std::string> tensor_spaces;
        tensor_spaces.reserve(tensor->indices().size());
        for (const auto& index : tensor->indices()) {
            const auto found_space = index_spaces.find(index);
            if (found_space == index_spaces.end()) {
                throw std::invalid_argument("index " + index + " does not belong to a space");
            }
            tensor_spaces.push_back(found_space->second);
        }
        const TensorArrayKey key{tensor->symbol(), tensor_spaces};
        const auto found_array = arrays.find(key);
        if (found_array == arrays.end()) {
            std::string spaces_text;
            for (const auto& space : tensor_spaces) {
                if (!spaces_text.empty()) {
                    spaces_text += ", ";
                }
                spaces_text += space;
            }
            throw std::invalid_argument("no array supplied for tensor " + tensor->symbol() +
                                        " in spaces (" + spaces_text + ")");
        }
        const auto& array = found_array->second;
        if (array.shape.size() != tensor->indices().size()) {
            throw std::invalid_argument("array rank does not match tensor " + tensor->symbol());
        }
        if (array.values.size() != checked_product(array.shape)) {
            throw std::invalid_argument("array data size does not match its shape for tensor " +
                                        tensor->symbol());
        }

        TensorFactor<T> prepared{&array, tensor->indices(), strides(array.shape)};
        for (std::size_t axis = 0; axis < tensor->indices().size(); ++axis) {
            const auto& index = tensor->indices()[axis];
            const auto& space = tensor_spaces[axis];
            if (array.shape[axis] != space_size(space, sizes)) {
                throw std::invalid_argument("array axis for tensor " + tensor->symbol() +
                                            " does not match declared size of space " + space);
            }
            const auto [dimension, inserted] = result.dimensions.emplace(index, array.shape[axis]);
            if (!inserted && dimension->second != array.shape[axis]) {
                throw std::invalid_argument("Einstein index " + index +
                                            " has inconsistent dimensions");
            }
            if (counts.find(index) == counts.end()) {
                index_order.push_back(index);
            }
            ++counts[index];
        }
        result.tensors.push_back(std::move(prepared));
    }

    for (const auto& index : index_order) {
        const auto count = counts.at(index);
        if (count == 1) {
            result.external_indices.push_back(index);
        } else if (count == 2) {
            result.internal_indices.push_back(index);
        } else {
            throw std::invalid_argument("Einstein index " + index +
                                        " occurs more than twice in a term");
        }
    }
    return result;
}

template <typename T>
DenseTensor<T> evaluate_rhs(const Expression& expression,
                            const TensorArrays<T>& arrays,
                            const IndexSpaceMap& index_spaces,
                            const SpaceSizes& sizes) {
    const auto& lhs = *expression.lhs();
    std::unordered_set<std::string> output_set;
    for (const auto& index : lhs.indices()) {
        if (!output_set.insert(index).second) {
            throw std::invalid_argument("an output tensor cannot repeat an index");
        }
        if (index_spaces.find(index) == index_spaces.end()) {
            throw std::invalid_argument("output index " + index + " does not belong to a space");
        }
    }

    std::vector<Monomial> monomials;
    expand_terms(expression.rhs(), monomials);
    std::vector<PreparedMonomial<T>> prepared;
    prepared.reserve(monomials.size());
    for (const auto& monomial : monomials) {
        auto term = prepare_monomial<T>(monomial, arrays, index_spaces, sizes);
        std::unordered_set<std::string> term_external(term.external_indices.begin(),
                                                      term.external_indices.end());
        if (term_external != output_set) {
            throw std::invalid_argument(
                "each expanded term must have the same external indices as the output tensor");
        }
        prepared.push_back(std::move(term));
    }

    std::vector<std::size_t> output_shape;
    output_shape.reserve(lhs.indices().size());
    for (const auto& index : lhs.indices()) {
        output_shape.push_back(space_size(index_spaces.at(index), sizes));
    }
    const auto output_count = checked_product(output_shape);
    if (output_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("output tensor has too many elements to evaluate");
    }
    DenseTensor<T> result{output_shape, std::vector<T>(output_count, T{})};
    const auto output_strides = strides(output_shape);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (output_count >= 64)
#endif
    for (std::ptrdiff_t flat_output = 0;
         flat_output < static_cast<std::ptrdiff_t>(output_count); ++flat_output) {
        auto remaining = static_cast<std::size_t>(flat_output);
        std::unordered_map<std::string, std::size_t> coordinates;
        for (std::size_t axis = 0; axis < lhs.indices().size(); ++axis) {
            const auto stride = output_strides[axis];
            const auto coordinate = stride == 0 ? 0 : remaining / stride;
            remaining = stride == 0 ? 0 : remaining % stride;
            coordinates.emplace(lhs.indices()[axis], coordinate);
        }

        T output_value{};
        for (const auto& term : prepared) {
            T term_value{};
            std::function<void(std::size_t)> sum_internal = [&](std::size_t axis) {
                if (axis < term.internal_indices.size()) {
                    const auto& index = term.internal_indices[axis];
                    const auto dimension = term.dimensions.at(index);
                    for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
                        coordinates[index] = coordinate;
                        sum_internal(axis + 1);
                    }
                    return;
                }
                T product = term.coefficient;
                for (const auto& tensor : term.tensors) {
                    std::size_t offset = 0;
                    for (std::size_t tensor_axis = 0;
                         tensor_axis < tensor.indices.size(); ++tensor_axis) {
                        offset += coordinates.at(tensor.indices[tensor_axis]) *
                                  tensor.strides[tensor_axis];
                    }
                    product *= tensor.array->values[offset];
                }
                term_value += product;
            };
            sum_internal(0);
            output_value += term_value;
        }
        result.values[static_cast<std::size_t>(flat_output)] = output_value;
    }
    return result;
}

template <typename T>
TensorArrays<T> parse_arrays(const py::dict& input) {
    TensorArrays<T> result;
    for (const auto& item : input) {
        if (!py::isinstance<py::tuple>(item.first)) {
            throw py::type_error("array keys must be (tensor_symbol, spaces_tuple) pairs");
        }
        const auto key = py::reinterpret_borrow<py::tuple>(item.first);
        if (key.size() != 2 || !py::isinstance<py::str>(key[0]) ||
            !py::isinstance<py::tuple>(key[1])) {
            throw py::type_error("array keys must be (tensor_symbol, spaces_tuple) pairs");
        }
        TensorArrayKey array_key{key[0].cast<std::string>(),
                                 key[1].cast<std::vector<std::string>>()};
        const auto raw_array = py::array::ensure(item.second);
        if (!raw_array) {
            throw py::type_error("tensor array values must be NumPy arrays");
        }
        const auto array =
            py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(raw_array);
        if (!array) {
            throw py::type_error("all tensor arrays must have the same supported dtype");
        }
        DenseTensor<T> tensor;
        tensor.shape.reserve(static_cast<std::size_t>(array.ndim()));
        for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
            tensor.shape.push_back(static_cast<std::size_t>(array.shape(axis)));
        }
        const auto count = checked_product(tensor.shape);
        if (count != 0) {
            tensor.values.assign(array.data(), array.data() + count);
        }
        if (!result.emplace(std::move(array_key), std::move(tensor)).second) {
            throw py::value_error("duplicate tensor-array key");
        }
    }
    return result;
}

template <typename T>
py::array_t<T> to_numpy(const DenseTensor<T>& tensor) {
    std::vector<py::ssize_t> shape;
    shape.reserve(tensor.shape.size());
    for (const auto dimension : tensor.shape) {
        if (dimension > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
            throw std::overflow_error("output dimension exceeds NumPy limits");
        }
        shape.push_back(static_cast<py::ssize_t>(dimension));
    }
    py::array_t<T> result(shape);
    std::copy(tensor.values.begin(), tensor.values.end(), result.mutable_data());
    return result;
}

template <typename T>
py::array_t<T> evaluate_from_python(const std::vector<Expression>& expressions,
                                    const py::dict& input,
                                    const py::object& metadata) {
    auto arrays = parse_arrays<T>(input);
    const auto spaces = metadata.attr("spaces").cast<IndexSpaces>();
    const auto sizes = metadata.attr("sizes").cast<SpaceSizes>();
    return to_numpy(evaluate_expression<T>(expressions, std::move(arrays), spaces, sizes));
}

}  // namespace

std::size_t TensorArrayKeyHash::operator()(const TensorArrayKey& key) const noexcept {
    std::size_t result = std::hash<std::string>{}(key.symbol);
    for (const auto& space : key.spaces) {
        const auto value = std::hash<std::string>{}(space);
        result ^= value + 0x9e3779b9U + (result << 6U) + (result >> 2U);
    }
    return result;
}

template <typename T>
DenseTensor<T> evaluate_expression(
    const std::vector<Expression>& expressions, TensorArrays<T> arrays,
    const IndexSpaces& spaces, const SpaceSizes& sizes) {
    if (expressions.empty()) {
        throw std::invalid_argument("evaluate_expression requires at least one expression");
    }
    const auto index_spaces = make_index_space_map(spaces);
    const auto graph = expression_dependency_graph(expressions);
    std::vector<std::string> outputs;
    for (std::size_t index = 0; index < expressions.size(); ++index) {
        if (graph.dependents[index].empty()) {
            outputs.push_back(expressions[index].lhs()->symbol());
        }
    }
    if (outputs.size() != 1) {
        throw std::invalid_argument("evaluate_expression requires exactly one terminal output");
    }
    const auto output_symbol = outputs.front();
    auto ordered = topological_sort(expressions);
    DenseTensor<T> result;
    for (const auto& expression : ordered) {
        auto evaluated = evaluate_rhs(expression, arrays, index_spaces, sizes);
        std::vector<std::string> lhs_spaces;
        lhs_spaces.reserve(expression.lhs()->indices().size());
        for (const auto& index : expression.lhs()->indices()) {
            const auto found = index_spaces.find(index);
            if (found == index_spaces.end()) {
                throw std::invalid_argument("output index " + index + " does not belong to a space");
            }
            lhs_spaces.push_back(found->second);
        }
        arrays.insert_or_assign(
            TensorArrayKey{expression.lhs()->symbol(), std::move(lhs_spaces)}, evaluated);
        if (expression.lhs()->symbol() == output_symbol) {
            result = std::move(evaluated);
        }
    }
    return result;
}

template DenseTensor<float> evaluate_expression(
    const std::vector<Expression>&, TensorArrays<float>, const IndexSpaces&, const SpaceSizes&);
template DenseTensor<double> evaluate_expression(
    const std::vector<Expression>&, TensorArrays<double>, const IndexSpaces&, const SpaceSizes&);
template DenseTensor<std::complex<float>> evaluate_expression(
    const std::vector<Expression>&, TensorArrays<std::complex<float>>, const IndexSpaces&,
    const SpaceSizes&);
template DenseTensor<std::complex<double>> evaluate_expression(
    const std::vector<Expression>&, TensorArrays<std::complex<double>>, const IndexSpaces&,
    const SpaceSizes&);

void bind_evaluate_expression(py::module_& module) {
    module.def(
        "evaluate_expression",
        [](const std::vector<Expression>& expressions, const py::dict& arrays,
           const py::object& metadata) -> py::object {
            if (metadata.is_none()) {
                throw py::type_error("metadata is required to map indices to spaces");
            }
            std::string dtype;
            for (const auto& item : arrays) {
                const auto array = py::array::ensure(item.second);
                if (!array) {
                    throw py::type_error("tensor array values must be NumPy arrays");
                }
                const auto name = py::str(array.dtype().attr("name")).cast<std::string>();
                if (dtype.empty()) {
                    dtype = name;
                } else if (dtype != name) {
                    throw py::type_error("all tensor arrays must have the same dtype");
                }
            }
            if (dtype.empty() || dtype == "float64") {
                return evaluate_from_python<double>(expressions, arrays, metadata);
            }
            if (dtype == "float32") {
                return evaluate_from_python<float>(expressions, arrays, metadata);
            }
            if (dtype == "complex64") {
                return evaluate_from_python<std::complex<float>>(expressions, arrays, metadata);
            }
            if (dtype == "complex128") {
                return evaluate_from_python<std::complex<double>>(expressions, arrays, metadata);
            }
            throw py::type_error("tensor arrays must use float32, float64, complex64, or complex128");
        },
        py::arg("expressions"), py::arg("arrays"), py::arg("metadata"),
        "Evaluate a single-output expression graph using NumPy tensor arrays.");
}

}  // namespace albert
