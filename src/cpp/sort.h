#pragma once

#include "expression.h"

#include <vector>

namespace albert {

// Uses input position to break ties between expressions currently ready to emit.
std::vector<Expression> topological_sort(std::vector<Expression> expressions);

}  // namespace albert
