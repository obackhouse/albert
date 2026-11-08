#pragma once

#include "ast.h"

namespace albert {

NodePtr expand(const NodePtr& node);
NodePtr collect(const NodePtr& node);
NodePtr squeeze(const NodePtr& node);
NodePtr remove_neutral_elements(const NodePtr& node);
NodePtr simplify(const NodePtr& node);

}  // namespace albert
