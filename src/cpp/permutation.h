#pragma once

#include "ast.h"

#include <cstdint>
#include <vector>

namespace albert {

enum class PhaseFactor : std::int8_t {
    Positive = 1,
    Negative = -1,
};

class Permutation {
public:
    explicit Permutation(std::vector<int> permutation,
                         PhaseFactor phase = PhaseFactor::Positive);

    const std::vector<int>& permutation() const noexcept;
    PhaseFactor phase() const noexcept;

private:
    std::vector<int> permutation_;
    PhaseFactor phase_;
};

NodePtr permute_indices(const NodePtr& node,
                        const std::vector<int>& permutation);
NodePtr permute_indices(const NodePtr& node,
                        const Permutation& permutation);

}  // namespace albert
