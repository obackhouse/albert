#pragma once

#include "ast.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace albert {

enum class SearchOrder : std::uint8_t {
    PreOrder,
    PostOrder,
};

namespace detail {

struct PredicateCallback {
    const void* state;
    bool (*invoke)(const void*, const NodePtr&);
};

struct TransformCallback {
    const void* state;
    NodePtr (*invoke)(const void*, const NodePtr&);
};

template <typename Callable>
PredicateCallback predicate_callback(Callable& callable) {
    using Callback = std::remove_reference_t<Callable>;
    return {
        &callable,
        [](const void* state, const NodePtr& node) -> bool {
            if constexpr (std::is_const_v<Callback>) {
                return (*static_cast<const Callback*>(state))(node);
            } else {
                return (*const_cast<Callback*>(static_cast<const Callback*>(state)))(node);
            }
        },
    };
}

template <typename Callable>
TransformCallback transform_callback(Callable& callable) {
    using Callback = std::remove_reference_t<Callable>;
    return {
        &callable,
        [](const void* state, const NodePtr& node) -> NodePtr {
            if constexpr (std::is_const_v<Callback>) {
                return (*static_cast<const Callback*>(state))(node);
            } else {
                return (*const_cast<Callback*>(static_cast<const Callback*>(state)))(node);
            }
        },
    };
}

std::vector<NodePtr> search_impl(const NodePtr& root, PredicateCallback predicate,
                                std::optional<std::size_t> max_depth,
                                SearchOrder order);
NodePtr find_impl(const NodePtr& root, PredicateCallback predicate,
                  std::optional<std::size_t> max_depth, SearchOrder order);
NodePtr delete_impl(const NodePtr& root, PredicateCallback predicate,
                    std::optional<std::size_t> max_depth);
NodePtr apply_impl(const NodePtr& root, TransformCallback filter,
                   PredicateCallback predicate,
                   std::optional<std::size_t> max_depth);

}  // namespace detail

template <typename Predicate>
std::vector<NodePtr> search(const NodePtr& root, Predicate&& predicate,
                            std::optional<std::size_t> max_depth = std::nullopt,
                            SearchOrder order = SearchOrder::PreOrder) {
    return detail::search_impl(
        root, detail::predicate_callback(predicate), max_depth, order);
}

template <typename Predicate>
NodePtr find(const NodePtr& root, Predicate&& predicate,
             std::optional<std::size_t> max_depth = std::nullopt,
             SearchOrder order = SearchOrder::PreOrder) {
    return detail::find_impl(
        root, detail::predicate_callback(predicate), max_depth, order);
}

template <typename Predicate>
NodePtr delete_nodes(const NodePtr& root, Predicate&& predicate,
                     std::optional<std::size_t> max_depth = std::nullopt) {
    return detail::delete_impl(
        root, detail::predicate_callback(predicate), max_depth);
}

template <typename Filter, typename Predicate>
NodePtr apply(const NodePtr& root, Filter&& filter, Predicate&& predicate,
              std::optional<std::size_t> max_depth = std::nullopt) {
    return detail::apply_impl(
        root, detail::transform_callback(filter),
        detail::predicate_callback(predicate), max_depth);
}

template <typename Filter>
NodePtr apply(const NodePtr& root, Filter&& filter,
              std::optional<std::size_t> max_depth = std::nullopt) {
    auto match_all = [](const NodePtr&) { return true; };
    return apply(root, std::forward<Filter>(filter), match_all, max_depth);
}

}  // namespace albert
