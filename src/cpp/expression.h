#pragma once

#include "ast.h"
#include "cost.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace albert {

class Expression {
public:
    Expression(std::shared_ptr<Tensor> lhs, NodePtr rhs);

    const std::shared_ptr<Tensor>& lhs() const noexcept;
    const NodePtr& rhs() const noexcept;

private:
    std::shared_ptr<Tensor> lhs_;
    NodePtr rhs_;
};

struct ExpressionDependencyGraph {
    // dependents[i] contains expressions whose RHS references expression i's LHS symbol.
    std::vector<std::vector<std::size_t>> dependents;
    // indegrees[i] is the number of distinct in-list definitions used by expression i.
    std::vector<std::size_t> indegrees;
};

// Treats RHS symbols with no definition in the list as external inputs.
ExpressionDependencyGraph expression_dependency_graph(
    const std::vector<Expression>& expressions);
// When spaces are provided, definitions and references are matched by symbol
// and ordered index-space tuple rather than symbol alone.
ExpressionDependencyGraph expression_dependency_graph(
    const std::vector<Expression>& expressions, const IndexSpaces& spaces);

}  // namespace albert
