#include "expression.h"

#include "algebra.h"

#include <algorithm>
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
    auto lhs_indices = external_indices(*lhs_);
    auto rhs_indices = external_indices(*rhs_);
    std::sort(lhs_indices.begin(), lhs_indices.end());
    std::sort(rhs_indices.begin(), rhs_indices.end());
    if (lhs_indices != rhs_indices) {
        throw std::invalid_argument(
            "Expression lhs and rhs must have the same external indices");
    }
}

const std::shared_ptr<Tensor>& Expression::lhs() const noexcept {
    return lhs_;
}

const NodePtr& Expression::rhs() const noexcept {
    return rhs_;
}

namespace {

std::string tensor_key(
    const Tensor& tensor,
    const std::unordered_map<std::string, std::string>* index_spaces) {
    if (!index_spaces) {
        return tensor.symbol();
    }
    std::string key = std::to_string(tensor.symbol().size()) + ":" + tensor.symbol() + ";" +
                      std::to_string(tensor.indices().size()) + ";";
    for (const auto& index : tensor.indices()) {
        const auto space = index_spaces->find(index);
        if (space == index_spaces->end()) {
            throw std::invalid_argument("index " + index + " does not belong to a metadata space");
        }
        key += std::to_string(space->second.size()) + ":" + space->second + ";";
    }
    return key;
}

void collect_tensor_keys(
    const NodePtr& node, std::unordered_set<std::string>& keys,
    const std::unordered_map<std::string, std::string>* index_spaces) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        keys.insert(tensor_key(*tensor, index_spaces));
        return;
    }
    for (const auto& child : node->children()) {
        collect_tensor_keys(child, keys, index_spaces);
    }
}

ExpressionDependencyGraph expression_dependency_graph_impl(
    const std::vector<Expression>& expressions,
    const std::unordered_map<std::string, std::string>* index_spaces) {
    std::unordered_map<std::string, std::size_t> definitions;
    definitions.reserve(expressions.size());
    for (std::size_t i = 0; i < expressions.size(); ++i) {
        const auto key = tensor_key(*expressions[i].lhs(), index_spaces);
        const auto [entry, inserted] = definitions.emplace(key, i);
        if (!inserted) {
            throw std::invalid_argument("duplicate expression definition for symbol " +
                                        expressions[i].lhs()->symbol());
        }
        static_cast<void>(entry);
    }

    ExpressionDependencyGraph graph;
    graph.dependents.resize(expressions.size());
    graph.indegrees.assign(expressions.size(), 0);
    for (std::size_t consumer = 0; consumer < expressions.size(); ++consumer) {
        std::unordered_set<std::string> keys;
        collect_tensor_keys(expressions[consumer].rhs(), keys, index_spaces);
        std::unordered_set<std::size_t> dependencies;
        for (const auto& key : keys) {
            const auto definition = definitions.find(key);
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

}  // namespace

ExpressionDependencyGraph expression_dependency_graph(
    const std::vector<Expression>& expressions) {
    return expression_dependency_graph_impl(expressions, nullptr);
}

ExpressionDependencyGraph expression_dependency_graph(
    const std::vector<Expression>& expressions, const IndexSpaces& spaces) {
    std::unordered_map<std::string, std::string> index_spaces;
    for (const auto& [space, indices] : spaces) {
        for (const auto& index : indices) {
            const auto [entry, inserted] = index_spaces.emplace(index, space);
            if (!inserted && entry->second != space) {
                throw std::invalid_argument("index " + index + " belongs to multiple metadata spaces");
            }
        }
    }
    return expression_dependency_graph_impl(expressions, &index_spaces);
}

}  // namespace albert
