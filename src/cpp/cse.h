#pragma once

#include "biclique.h"
#include "canonicalize.h"
#include "cost.h"
#include "expression.h"
#include "parenthesize.h"

#include <cstddef>
#include <string>
#include <vector>

namespace albert {

struct CSEOptions {
    std::string intermediate_format = "__im{}__";
    std::size_t max_path_samples = 19;
    std::size_t max_bicliques = 8;
    std::size_t exhaustive_limit = 4;
    bool custom_path_requests = false;
    PathRequests path_requests{{4}, {{18, 0, 0, true}, {0, 0, 0, true}}};
    ScalingLimits scaling_limits;
    CostModel cost_model;
};

std::vector<Expression> common_subexpression_elimination(
    std::vector<Expression> expressions, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostFactors& cost_factors,
    const CanonicalizationMetadata& canonicalization = {},
    const CSEOptions& options = {});

}  // namespace albert
