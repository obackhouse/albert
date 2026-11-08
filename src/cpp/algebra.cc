#include "algebra.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace albert {
namespace {

void validate_children(const Children& children) {
    if (children.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("node arity exceeds uint32_t capacity");
    }
    for (const auto& child : children) {
        if (!child) {
            throw std::invalid_argument("AST children cannot be null");
        }
    }
}

std::vector<std::string> free_indices(const Node& node) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(&node)) {
        std::unordered_map<std::string, std::size_t> counts;
        for (const auto& index : tensor->indices()) {
            ++counts[index];
        }
        std::vector<std::string> result;
        for (const auto& index : tensor->indices()) {
            if (counts[index] == 1) {
                result.push_back(index);
                counts[index] = 0;
            }
        }
        return result;
    }
    if (const auto* add = dynamic_cast<const Add*>(&node)) {
        if (add->children().empty()) {
            return {};
        }
        return free_indices(*add->children().front());
    }
    if (const auto* mul = dynamic_cast<const Mul*>(&node)) {
        std::unordered_map<std::string, std::size_t> counts;
        std::vector<std::string> order;
        for (const auto& child : mul->children()) {
            for (const auto& index : free_indices(*child)) {
                if (counts.find(index) == counts.end()) {
                    order.push_back(index);
                }
                ++counts[index];
            }
        }
        std::vector<std::string> result;
        for (const auto& index : order) {
            if (counts[index] == 1) {
                result.push_back(index);
            }
        }
        return result;
    }
    return {};
}

void validate_add_indices(const Children& children) {
    if (children.empty()) {
        return;
    }
    auto expected = free_indices(*children.front());
    std::sort(expected.begin(), expected.end());
    for (std::size_t position = 1; position < children.size(); ++position) {
        auto current = free_indices(*children[position]);
        std::sort(current.begin(), current.end());
        if (current != expected) {
            throw std::invalid_argument("Add operands must have the same free indices");
        }
    }
}

}  // namespace

Add::Add(Children children)
    : Node(NodeKind::Add), children_(std::move(children)) {
    validate_children(children_);
    validate_add_indices(children_);
}

std::uint32_t Add::arity() const noexcept {
    return static_cast<std::uint32_t>(children_.size());
}

const Children& Add::children() const noexcept {
    return children_;
}

Mul::Mul(Children children)
    : Node(NodeKind::Mul), children_(std::move(children)) {
    validate_children(children_);
}

std::uint32_t Mul::arity() const noexcept {
    return static_cast<std::uint32_t>(children_.size());
}

const Children& Mul::children() const noexcept {
    return children_;
}

NodePtr operator+(NodePtr lhs, NodePtr rhs) {
    return std::make_shared<Add>(Children{std::move(lhs), std::move(rhs)});
}

NodePtr operator+(NodePtr lhs, double rhs) {
    return std::move(lhs) + std::make_shared<Scalar>(rhs);
}

NodePtr operator+(double lhs, NodePtr rhs) {
    return std::make_shared<Scalar>(lhs) + std::move(rhs);
}

NodePtr operator-(NodePtr lhs, NodePtr rhs) {
    return std::move(lhs) + (-std::move(rhs));
}

NodePtr operator-(NodePtr lhs, double rhs) {
    return std::move(lhs) + (-std::make_shared<Scalar>(rhs));
}

NodePtr operator-(double lhs, NodePtr rhs) {
    return std::make_shared<Scalar>(lhs) + (-std::move(rhs));
}

NodePtr operator*(NodePtr lhs, NodePtr rhs) {
    return std::make_shared<Mul>(Children{std::move(lhs), std::move(rhs)});
}

NodePtr operator*(NodePtr lhs, double rhs) {
    return std::move(lhs) * std::make_shared<Scalar>(rhs);
}

NodePtr operator*(double lhs, NodePtr rhs) {
    return std::make_shared<Scalar>(lhs) * std::move(rhs);
}

NodePtr operator+(NodePtr value) {
    return std::make_shared<Mul>(
        Children{std::make_shared<Scalar>(1.0), std::move(value)});
}

NodePtr operator-(NodePtr value) {
    return std::make_shared<Mul>(
        Children{std::make_shared<Scalar>(-1.0), std::move(value)});
}

}  // namespace albert
