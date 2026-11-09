#include "substitute.h"

#include "algebra.h"
#include "sort.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace albert {
namespace {

using IndexMapping = std::unordered_map<std::string, std::string>;

NodePtr map_all_indices(const NodePtr& node, const IndexMapping& mapping) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        auto indices = tensor->indices();
        bool changed = false;
        for (auto& index : indices) {
            const auto found = mapping.find(index);
            if (found != mapping.end() && found->second != index) {
                index = found->second;
                changed = true;
            }
        }
        return changed ? std::make_shared<Tensor>(tensor->symbol(), std::move(indices))
                       : node;
    }
    if (node->children().empty()) {
        return node;
    }
    Children children;
    children.reserve(node->children().size());
    bool changed = false;
    for (const auto& child : node->children()) {
        auto mapped = map_all_indices(child, mapping);
        changed = changed || mapped != child;
        children.push_back(std::move(mapped));
    }
    if (!changed) {
        return node;
    }
    if (node->kind() == NodeKind::Add) {
        return std::make_shared<Add>(std::move(children));
    }
    return std::make_shared<Mul>(std::move(children));
}

void collect_indices(const NodePtr& node, std::unordered_set<std::string>& indices) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        indices.insert(tensor->indices().begin(), tensor->indices().end());
    }
    for (const auto& child : node->children()) {
        collect_indices(child, indices);
    }
}

NodePtr substitute_node(const NodePtr& node,
                        const std::unordered_map<std::string, const Expression*>& definitions,
                        std::unordered_set<std::string>& used_indices,
                        std::size_t& fresh_index,
                        bool& changed) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        const auto found = definitions.find(tensor->symbol());
        if (found == definitions.end()) {
            return node;
        }
        const auto& definition = *found->second;
        if (definition.lhs()->rank() != tensor->rank()) {
            throw std::invalid_argument("rank mismatch while substituting expression " +
                                        tensor->symbol());
        }

        IndexMapping mapping;
        for (std::size_t axis = 0; axis < tensor->indices().size(); ++axis) {
            const auto& source = definition.lhs()->indices()[axis];
            const auto& target = tensor->indices()[axis];
            const auto [entry, inserted] = mapping.emplace(source, target);
            if (!inserted && entry->second != target) {
                throw std::invalid_argument(
                    "cannot map repeated expression index to distinct indices");
            }
        }

        // Internal indices belong to this definition's contraction. Give each
        // inserted copy fresh labels so names in the consumer cannot capture them.
        for (const auto& index : definition.rhs()->internal_indices()) {
            if (mapping.find(index) != mapping.end() ||
                used_indices.find(index) == used_indices.end()) {
                continue;
            }
            std::string fresh;
            do {
                fresh = "__substitute_" + std::to_string(fresh_index++);
            } while (used_indices.find(fresh) != used_indices.end());
            mapping[index] = fresh;
            used_indices.insert(fresh);
        }
        changed = true;
        auto result = map_all_indices(definition.rhs(), mapping);
        collect_indices(result, used_indices);
        return result;
    }

    if (node->children().empty()) {
        return node;
    }
    Children children;
    children.reserve(node->children().size());
    bool child_changed = false;
    for (const auto& child : node->children()) {
        auto substituted = substitute_node(child, definitions, used_indices,
                                           fresh_index, child_changed);
        children.push_back(std::move(substituted));
    }
    if (!child_changed) {
        return node;
    }
    changed = true;
    if (node->kind() == NodeKind::Add) {
        return std::make_shared<Add>(std::move(children));
    }
    return std::make_shared<Mul>(std::move(children));
}

}  // namespace

std::vector<Expression> substitute_expressions(std::vector<Expression> expressions) {
    // Sorting validates unique definitions and cycles. The sorted list gives us
    // an order in which all dependencies of a definition have already expanded.
    auto ordered = topological_sort(std::move(expressions));
    std::unordered_set<std::string> removable;
    for (const auto& definition : ordered) {
        const auto symbol = definition.lhs()->symbol();
        std::unordered_map<std::string, const Expression*> candidate{
            {symbol, &definition}};
        for (auto& consumer : ordered) {
            if (&consumer == &definition) {
                continue;
            }
            std::unordered_set<std::string> occupied;
            collect_indices(consumer.lhs(), occupied);
            collect_indices(consumer.rhs(), occupied);
            std::size_t fresh_index = 0;
            bool changed = false;
            auto rhs = substitute_node(consumer.rhs(), candidate, occupied,
                                       fresh_index, changed);
            if (changed) {
                consumer = Expression(consumer.lhs(), std::move(rhs));
                removable.insert(symbol);
            }
        }
    }

    std::vector<Expression> result;
    result.reserve(ordered.size() - removable.size());
    for (auto& expression : ordered) {
        if (removable.find(expression.lhs()->symbol()) == removable.end()) {
            result.push_back(std::move(expression));
        }
    }
    return result;
}

}  // namespace albert
