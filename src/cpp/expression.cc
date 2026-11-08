#include "expression.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace albert {

Expression::Expression(std::shared_ptr<Tensor> lhs, NodePtr rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    if (!lhs_) {
        throw std::invalid_argument("Expression lhs cannot be null");
    }
    if (!rhs_) {
        throw std::invalid_argument("Expression rhs cannot be null");
    }
}

const std::shared_ptr<Tensor>& Expression::lhs() const noexcept {
    return lhs_;
}

const NodePtr& Expression::rhs() const noexcept {
    return rhs_;
}

namespace {

void collect_tensor_symbols(const NodePtr& node,
                            std::unordered_set<std::string>& symbols) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        symbols.insert(tensor->symbol());
        return;
    }
    for (const auto& child : node->children()) {
        collect_tensor_symbols(child, symbols);
    }
}

}  // namespace

ExpressionDependencyGraph expression_dependency_graph(
    const std::vector<Expression>& expressions) {
    std::unordered_map<std::string, std::size_t> definitions;
    definitions.reserve(expressions.size());
    for (std::size_t i = 0; i < expressions.size(); ++i) {
        const auto& symbol = expressions[i].lhs()->symbol();
        const auto [entry, inserted] = definitions.emplace(symbol, i);
        if (!inserted) {
            throw std::invalid_argument("duplicate expression definition for symbol " +
                                        symbol);
        }
        static_cast<void>(entry);
    }

    ExpressionDependencyGraph graph;
    graph.dependents.resize(expressions.size());
    graph.indegrees.assign(expressions.size(), 0);
    for (std::size_t consumer = 0; consumer < expressions.size(); ++consumer) {
        std::unordered_set<std::string> symbols;
        collect_tensor_symbols(expressions[consumer].rhs(), symbols);
        std::unordered_set<std::size_t> dependencies;
        for (const auto& symbol : symbols) {
            const auto definition = definitions.find(symbol);
            if (definition != definitions.end()) {
                dependencies.insert(definition->second);
            }
        }
        for (const auto dependency : dependencies) {
            graph.dependents[dependency].push_back(consumer);
            ++graph.indegrees[consumer];
        }
    }
    return graph;
}

}  // namespace albert
