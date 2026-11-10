#pragma once

#include "expression.h"
#include "cost.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pybind11 {
class module_;
}

namespace albert {

struct TensorArrayKey {
    std::string symbol;
    std::vector<std::string> spaces;

    bool operator==(const TensorArrayKey& other) const noexcept {
        return symbol == other.symbol && spaces == other.spaces;
    }
};

struct TensorArrayKeyHash {
    std::size_t operator()(const TensorArrayKey& key) const noexcept;
};

template <typename T> struct DenseTensor {
    std::vector<std::size_t> shape;
    std::vector<T> values;
};

template <typename T>
using TensorArrays = std::unordered_map<TensorArrayKey, DenseTensor<T>,
                                        TensorArrayKeyHash>;

template <typename T>
DenseTensor<T> evaluate_expression(
    const std::vector<Expression>& expressions, TensorArrays<T> arrays,
    const IndexSpaces& spaces, const SpaceSizes& sizes);

void bind_evaluate_expression(pybind11::module_& module);

}  // namespace albert
