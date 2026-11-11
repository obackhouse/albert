#include "sort.h"

#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace albert {
namespace {

std::vector<Expression> sort_with_graph(
    std::vector<Expression> expressions, ExpressionDependencyGraph graph) {
    using ReadyQueue = std::priority_queue<std::size_t,
                                           std::vector<std::size_t>,
                                           std::greater<std::size_t>>;
    ReadyQueue ready;
    for (std::size_t i = 0; i < graph.indegrees.size(); ++i) {
        if (graph.indegrees[i] == 0) {
            ready.push(i);
        }
    }

    std::vector<Expression> ordered;
    ordered.reserve(expressions.size());
    while (!ready.empty()) {
        const auto next = ready.top();
        ready.pop();
        ordered.push_back(std::move(expressions[next]));
        for (const auto dependent : graph.dependents[next]) {
            if (--graph.indegrees[dependent] == 0) {
                ready.push(dependent);
            }
        }
    }
    if (ordered.size() != expressions.size()) {
        throw std::invalid_argument("cycle detected in expression dependencies");
    }
    return ordered;
}

}  // namespace

std::vector<Expression> topological_sort(std::vector<Expression> expressions) {
    auto graph = expression_dependency_graph(expressions);
    return sort_with_graph(std::move(expressions), std::move(graph));
}

std::vector<Expression> topological_sort(std::vector<Expression> expressions,
                                         const IndexSpaces& spaces) {
    auto graph = expression_dependency_graph(expressions, spaces);
    return sort_with_graph(std::move(expressions), std::move(graph));
}

}  // namespace albert
