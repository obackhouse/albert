#pragma once

#include "ast.h"
#include "permutation.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace albert {

struct CanonicalizationMetadata {
    std::unordered_map<std::string, std::vector<std::string>> spaces;
    std::unordered_map<std::string, std::vector<Permutation>> symmetries;
};

NodePtr canonicalize(const NodePtr& node,
                     const CanonicalizationMetadata& metadata = {});

}  // namespace albert
