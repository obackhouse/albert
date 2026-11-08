#include "permutation.h"

#include "algebra.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace albert {

Permutation::Permutation(std::vector<int> permutation, PhaseFactor phase)
    : permutation_(std::move(permutation)), phase_(phase) {
    if (phase_ != PhaseFactor::Positive && phase_ != PhaseFactor::Negative) {
        throw std::invalid_argument("permutation phase must be positive or negative");
    }

    std::vector<bool> seen(permutation_.size(), false);
    for (const auto index : permutation_) {
        if (index < 0 || static_cast<std::size_t>(index) >= permutation_.size() ||
            seen[static_cast<std::size_t>(index)]) {
            throw std::invalid_argument("permutation must contain each index from 0 to n-1 once");
        }
        seen[static_cast<std::size_t>(index)] = true;
    }
}

const std::vector<int>& Permutation::permutation() const noexcept {
    return permutation_;
}

PhaseFactor Permutation::phase() const noexcept {
    return phase_;
}

NodePtr permute_indices(const NodePtr& node,
                        const std::vector<int>& permutation) {
    return permute_indices(node, Permutation(permutation));
}

NodePtr permute_indices(const NodePtr& node,
                        const Permutation& permutation) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
    const auto indices = node->external_indices();
    if (permutation.permutation().size() != indices.size()) {
        throw std::invalid_argument(
            "permutation length must match the number of external indices");
    }

    std::unordered_map<std::string, std::string> mapping;
    mapping.reserve(indices.size());
    for (std::size_t position = 0; position < indices.size(); ++position) {
        const auto source = indices[position];
        const auto target = indices[static_cast<std::size_t>(
            permutation.permutation()[position])];
        mapping.emplace(source, target);
    }
    return albert::map_indices(node, mapping);
}

}  // namespace albert
