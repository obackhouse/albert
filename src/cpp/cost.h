#pragma once

#include "ast.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace albert {

using IndexSpaces = std::unordered_map<std::string, std::vector<std::string>>;
using SpaceSizes = std::unordered_map<std::string, std::uint64_t>;

struct CostModel {
    double multiply = 1.0;
    double add = 1.0;
    double read = 0.1;
    double write = 0.1;
};

struct CostEstimate {
    double multiplications = 0.0;
    double additions = 0.0;
    double reads = 0.0;
    double writes = 0.0;
    double total = 0.0;
};

CostEstimate estimate_cost(const NodePtr& node, const IndexSpaces& spaces,
                           const SpaceSizes& sizes,
                           const CostModel& model = {});

std::size_t scaling_exponent(const NodePtr& node, const IndexSpaces& spaces,
                             const std::vector<std::string>& selected_spaces,
                             const std::string& which);

}  // namespace albert
