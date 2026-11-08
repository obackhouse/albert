#include "ast.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace albert {

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
