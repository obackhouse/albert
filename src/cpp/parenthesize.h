#pragma once

#include "algebra.h"
#include "cost.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace albert {

using Contraction = std::pair<std::size_t, std::size_t>;

struct ParenthesizationCandidate {
    std::vector<Contraction> path;
    CostEstimate cost;
};

using ScalingCaps = std::vector<std::pair<std::vector<std::string>, std::size_t>>;

struct ScalingLimits {
    ScalingCaps time;
    ScalingCaps memory;
};

struct PathRequest {
    std::size_t exhaustive = 0;
    std::size_t branch = 0;
    std::size_t greedy = 0;
    bool dynamic = false;
};

struct PathRequests {
    std::vector<std::size_t> bounds;
    std::vector<PathRequest> requests;

    const PathRequest& for_size(std::size_t operand_count) const;
};

std::vector<ParenthesizationCandidate> parenthesization_candidates(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits = {}, std::size_t max_candidates = 8,
    std::size_t exhaustive_limit = 5, const CostModel& model = {},
    const CostFactors& cost_factors = {});

std::vector<ParenthesizationCandidate> parenthesization_candidates(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits, const PathRequest& request,
    std::size_t max_candidates, const CostModel& model = {},
    const CostFactors& cost_factors = {});

NodePtr apply_parenthesization(const Mul& node,
                               const ParenthesizationCandidate& candidate);

}  // namespace albert
