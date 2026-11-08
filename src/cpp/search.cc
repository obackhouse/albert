#include "search.h"

#include "algebra.h"

#include <stdexcept>
#include <utility>

namespace albert::detail {
namespace {

void require_root(const NodePtr& root) {
    if (!root) {
        throw std::invalid_argument("AST root cannot be null");
    }
}

void search_node(const NodePtr& node, std::size_t depth,
                 PredicateCallback predicate,
                 std::optional<std::size_t> max_depth, SearchOrder order,
                 std::vector<NodePtr>& matches) {
    if (order == SearchOrder::PreOrder && predicate.invoke(predicate.state, node)) {
        matches.push_back(node);
    }
    if (!max_depth || depth < *max_depth) {
        for (const auto& child : node->children()) {
            search_node(child, depth + 1, predicate, max_depth, order, matches);
        }
    }
    if (order == SearchOrder::PostOrder && predicate.invoke(predicate.state, node)) {
        matches.push_back(node);
    }
}

NodePtr find_node(const NodePtr& node, std::size_t depth,
                  PredicateCallback predicate,
                  std::optional<std::size_t> max_depth, SearchOrder order) {
    if (order == SearchOrder::PreOrder && predicate.invoke(predicate.state, node)) {
        return node;
    }
    if (!max_depth || depth < *max_depth) {
        for (const auto& child : node->children()) {
            if (auto match = find_node(child, depth + 1, predicate, max_depth, order)) {
                return match;
            }
        }
    }
    if (order == SearchOrder::PostOrder && predicate.invoke(predicate.state, node)) {
        return node;
    }
    return nullptr;
}

NodePtr rebuild(const NodePtr& node, Children children) {
    if (node->kind() == NodeKind::Add) {
        return std::make_shared<Add>(std::move(children));
    }
    if (node->kind() == NodeKind::Mul) {
        return std::make_shared<Mul>(std::move(children));
    }
    throw std::logic_error("leaf node unexpectedly has children");
}

NodePtr apply_node(const NodePtr& node, std::size_t depth,
                   TransformCallback filter, PredicateCallback predicate,
                   std::optional<std::size_t> max_depth) {
    if (max_depth && depth > *max_depth) {
        return node;
    }

    NodePtr current = node;
    const auto& original_children = node->children();
    if (!original_children.empty() && (!max_depth || depth < *max_depth)) {
        Children children;
        bool changed = false;
        for (std::size_t position = 0; position < original_children.size(); ++position) {
            const auto& child = original_children[position];
            auto updated = apply_node(child, depth + 1, filter, predicate, max_depth);
            if (!changed && updated != child) {
                children.reserve(original_children.size());
                children.insert(children.end(), original_children.begin(),
                                original_children.begin() +
                                    static_cast<std::ptrdiff_t>(position));
                changed = true;
            }
            if (changed) {
                children.push_back(std::move(updated));
            }
        }
        if (changed) {
            current = rebuild(node, std::move(children));
        }
    }

    if (predicate.invoke(predicate.state, current)) {
        auto replacement = filter.invoke(filter.state, current);
        if (!replacement) {
            throw std::invalid_argument("apply filter cannot return a null node");
        }
        return replacement;
    }
    return current;
}

}  // namespace

std::vector<NodePtr> search_impl(const NodePtr& root,
                                 PredicateCallback predicate,
                                 std::optional<std::size_t> max_depth,
                                 SearchOrder order) {
    require_root(root);
    std::vector<NodePtr> matches;
    search_node(root, 0, predicate, max_depth, order, matches);
    return matches;
}

NodePtr find_impl(const NodePtr& root, PredicateCallback predicate,
                  std::optional<std::size_t> max_depth, SearchOrder order) {
    require_root(root);
    return find_node(root, 0, predicate, max_depth, order);
}

NodePtr delete_impl(const NodePtr& root, PredicateCallback predicate,
                    std::optional<std::size_t> max_depth) {
    auto delete_match = [](const NodePtr&) {
        return std::make_shared<Scalar>(0.0);
    };
    return apply_impl(root, transform_callback(delete_match), predicate, max_depth);
}

NodePtr apply_impl(const NodePtr& root, TransformCallback filter,
                   PredicateCallback predicate,
                   std::optional<std::size_t> max_depth) {
    require_root(root);
    return apply_node(root, 0, filter, predicate, max_depth);
}

}  // namespace albert::detail
