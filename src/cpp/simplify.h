#pragma once

#include "ast.h"

namespace albert {

NodePtr expand(const NodePtr& node);
NodePtr flatten(const NodePtr& node);
NodePtr collect(const NodePtr& node);
NodePtr squeeze(const NodePtr& node);
NodePtr prune(const NodePtr& node);
NodePtr simplify(const NodePtr& node);

}  // namespace albert
