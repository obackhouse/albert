#pragma once

#include "ast.h"
#include "permutation.h"
#include "tensor_key.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace albert {

struct CanonicalizationMetadata {
    std::unordered_map<std::string, std::vector<std::string>> spaces;
    std::unordered_map<TensorKey, std::vector<Permutation>, TensorKeyHash> symmetries;
};

struct CanonicalizedNode {
    NodePtr node;
    std::unordered_map<std::string, std::string> external_index_mapping;
};

NodePtr canonicalize(const NodePtr& node,
                     const CanonicalizationMetadata& metadata = {});
CanonicalizedNode canonicalize_with_mapping(
    const NodePtr& node, const CanonicalizationMetadata& metadata = {});

}  // namespace albert
