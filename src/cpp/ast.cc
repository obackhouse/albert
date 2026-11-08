#include "ast.h"

#include "algebra.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace albert {
namespace {

NodePtr map_indices_impl(
    const NodePtr& node,
    const std::unordered_map<std::string, std::string>& mapping,
    std::unordered_map<const Node*, NodePtr>& mapped_nodes) {
    const auto cached = mapped_nodes.find(node.get());
    if (cached != mapped_nodes.end()) {
        return cached->second;
    }

    NodePtr mapped = node;
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        auto indices = tensor->indices();
        bool changed = false;
        for (auto& index : indices) {
            const auto replacement = mapping.find(index);
            if (replacement != mapping.end() && replacement->second != index) {
                index = replacement->second;
                changed = true;
            }
        }
        if (changed) {
            mapped = std::make_shared<Tensor>(tensor->symbol(), std::move(indices));
        }
    } else if (!node->children().empty()) {
        Children children;
        children.reserve(node->children().size());
        bool changed = false;
        for (const auto& child : node->children()) {
            auto mapped_child = map_indices_impl(child, mapping, mapped_nodes);
            changed = changed || mapped_child != child;
            children.push_back(std::move(mapped_child));
        }
        if (changed) {
            if (node->kind() == NodeKind::Add) {
                mapped = std::make_shared<Add>(std::move(children));
            } else {
                mapped = std::make_shared<Mul>(std::move(children));
            }
        }
    }

    mapped_nodes.emplace(node.get(), mapped);
    return mapped;
}

}  // namespace

Node::Node(NodeKind kind) noexcept : kind_(kind) {}

NodeKind Node::kind() const noexcept {
    return kind_;
}

const Children& Node::children() const noexcept {
    static const Children empty;
    return empty;
}

Scalar::Scalar(double value) : Node(NodeKind::Scalar), value_(value) {}

double Scalar::value() const noexcept {
    return value_;
}

Tensor::Tensor(std::string symbol, std::vector<std::string> indices)
    : Node(NodeKind::Tensor), symbol_(std::move(symbol)),
      indices_(std::move(indices)) {
    if (indices_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("tensor rank exceeds uint32_t capacity");
    }
}

const std::string& Tensor::symbol() const noexcept {
    return symbol_;
}

std::uint32_t Tensor::rank() const noexcept {
    return static_cast<std::uint32_t>(indices_.size());
}

const std::vector<std::string>& Tensor::indices() const noexcept {
    return indices_;
}

NodePtr map_indices(
    const NodePtr& node,
    const std::unordered_map<std::string, std::string>& mapping) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
    if (mapping.empty()) {
        return node;
    }
    const auto external = node->external_indices();
    for (const auto& [index, replacement] : mapping) {
        static_cast<void>(replacement);
        if (std::find(external.begin(), external.end(), index) == external.end()) {
            throw std::invalid_argument(
                "map_indices keys must be external indices of the node");
        }
    }
    std::unordered_map<const Node*, NodePtr> mapped_nodes;
    return map_indices_impl(node, mapping, mapped_nodes);
}

}  // namespace albert
