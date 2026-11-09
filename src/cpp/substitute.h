#pragma once

#include "expression.h"
#include "cost.h"

#include <vector>

namespace albert {

std::vector<Expression> substitute_expressions(
    std::vector<Expression> expressions, const IndexSpaces& spaces);

}  // namespace albert
