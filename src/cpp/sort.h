#pragma once

#include "expression.h"

#include <vector>

namespace albert {

// Uses input position to break ties between expressions currently ready to emit.
std::vector<Expression> topological_sort(std::vector<Expression> expressions);
// Uses tensor symbols and ordered index-space tuples to identify dependencies.
std::vector<Expression> topological_sort(std::vector<Expression> expressions,
                                         const IndexSpaces& spaces);

}  // namespace albert
