#include "permutation.h"

#include <stdexcept>
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

}  // namespace albert
