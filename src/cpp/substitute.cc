#include "substitute.h"

#include "algebra.h"
#include "sort.h"

#include <algorithm>
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
using IndexSpaceMap = std::unordered_map<std::string, std::string>;

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
                        const IndexSpaces& spaces,
                        const IndexSpaceMap& index_spaces,
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
            const auto source_space = index_spaces.find(source);
            const auto target_space = index_spaces.find(target);
            if (source_space != index_spaces.end() && target_space != index_spaces.end() &&
                source_space->second != target_space->second) {
                throw std::invalid_argument(
                    "cannot substitute an expression across different index spaces");
            }
            const auto [entry, inserted] = mapping.emplace(source, target);
            if (!inserted && entry->second != target) {
                throw std::invalid_argument(
                    "cannot map repeated expression index to distinct indices");
            }
        }

        // Internal indices belong to this definition's contraction. Add nodes
        // report their external indices from a representative child, so collect
        // all RHS labels and exclude the equation's free indices explicitly.
        const auto external = definition.rhs()->external_indices();
        const std::unordered_set<std::string> external_set(external.begin(), external.end());
        std::unordered_set<std::string> rhs_indices;
        collect_indices(definition.rhs(), rhs_indices);
        for (const auto& index : rhs_indices) {
            if (external_set.find(index) != external_set.end()) {
                continue;
            }
            if (mapping.find(index) != mapping.end() ||
                used_indices.find(index) == used_indices.end()) {
                continue;
            }
            const auto owner = index_spaces.find(index);
            if (owner == index_spaces.end()) {
                throw std::invalid_argument(
                    "cannot determine the index space for substituted index " + index);
            }
            const auto labels = spaces.find(owner->second);
            std::vector<std::string> candidates =
                labels == spaces.end() ? std::vector<std::string>{} : labels->second;
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            std::string fresh;
            for (const auto& candidate : candidates) {
                if (used_indices.find(candidate) == used_indices.end() &&
                    rhs_indices.find(candidate) == rhs_indices.end()) {
                    fresh = candidate;
                    break;
                }
            }
            if (fresh.empty()) {
                do {
                    fresh = "__internal_" + owner->second + "_" +
                            std::to_string(fresh_index++) + "__";
                } while (used_indices.find(fresh) != used_indices.end() ||
                         rhs_indices.find(fresh) != rhs_indices.end());
            }
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
        auto substituted = substitute_node(child, definitions, spaces, index_spaces, used_indices,
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

std::vector<Expression> substitute_expressions(std::vector<Expression> expressions,
                                               const IndexSpaces& spaces) {
    // Sorting validates unique definitions and cycles. The sorted list gives us
    // an order in which all dependencies of a definition have already expanded.
    auto ordered = topological_sort(std::move(expressions));
    IndexSpaceMap index_spaces;
    for (const auto& [space, labels] : spaces) {
        for (const auto& label : labels) {
            const auto [found, inserted] = index_spaces.emplace(label, space);
            if (!inserted && found->second != space) {
                throw std::invalid_argument("an index cannot belong to multiple spaces");
            }
        }
    }
    std::unordered_set<std::string> removable;
    for (const auto& definition : ordered) {
        const auto symbol = definition.lhs()->symbol();
        std::unordered_map<std::string, const Expression*> candidate{
            {symbol, &definition}};
        for (auto& consumer : ordered) {
            if (&consumer == &definition) {
                continue;
            }
            bool changed = false;
            NodePtr rhs = consumer.rhs();
            if (rhs->kind() == NodeKind::Add) {
                Children terms;
                terms.reserve(rhs->children().size());
                bool changed_any = false;
                for (const auto& term : rhs->children()) {
                    std::unordered_set<std::string> occupied;
                    collect_indices(consumer.lhs(), occupied);
                    collect_indices(term, occupied);
                    std::size_t fresh_index = 0;
                    bool term_changed = false;
                    terms.push_back(substitute_node(term, candidate, spaces, index_spaces,
                                                    occupied, fresh_index, term_changed));
                    changed_any = changed_any || term_changed;
                }
                if (changed_any) {
                    rhs = std::make_shared<Add>(std::move(terms));
                }
                changed = changed_any;
            } else {
                std::unordered_set<std::string> occupied;
                collect_indices(consumer.lhs(), occupied);
                collect_indices(consumer.rhs(), occupied);
                std::size_t fresh_index = 0;
                rhs = substitute_node(consumer.rhs(), candidate, spaces, index_spaces,
                                      occupied, fresh_index, changed);
            }
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
