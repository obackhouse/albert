#include "ast.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace albert {
namespace {

void append_size(std::string& key, std::size_t value) {
    key += std::to_string(value);
    key.push_back(':');
}

void append_string(std::string& key, const std::string& value) {
    append_size(key, value.size());
    key.append(value);
}

void append_double(std::string& key, double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "canonical scalar keys require 64-bit doubles");
    std::uint64_t bits = 0;
    if (value != 0.0) {
        std::memcpy(&bits, &value, sizeof(bits));
    }
    constexpr char digits[] = "0123456789abcdef";
    for (int shift = 60; shift >= 0; shift -= 4) {
        key.push_back(digits[(bits >> shift) & 0x0f]);
    }
}

void append_node_key(const NodePtr& node, std::string& key) {
    if (const auto* scalar = dynamic_cast<const Scalar*>(node.get())) {
        key.push_back('S');
        append_double(key, scalar->value());
        return;
    }
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        key.push_back('T');
        append_string(key, tensor->symbol());
        append_size(key, tensor->indices().size());
        for (const auto& index : tensor->indices()) {
            append_string(key, index);
        }
        return;
    }

    key.push_back(node->kind() == NodeKind::Add ? 'A' : 'M');
    append_size(key, node->children().size());
    for (const auto& child : node->children()) {
        append_node_key(child, key);
    }
}

}  // namespace

std::string canonical_key(const NodePtr& node) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
    std::string key;
    append_node_key(node, key);
    return key;
}

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

}  // namespace albert
