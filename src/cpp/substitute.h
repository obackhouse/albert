#pragma once

#include "expression.h"

#include <vector>

namespace albert {

std::vector<Expression> substitute_expressions(
    std::vector<Expression> expressions);

}  // namespace albert
