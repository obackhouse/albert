#pragma once

#include "ast.h"

#include <cstdint>

namespace albert {

class Add final : public Node {
public:
    explicit Add(Children children);

    std::uint32_t arity() const noexcept;
    const Children& children() const noexcept override;

private:
    Children children_;
};

class Mul final : public Node {
public:
    explicit Mul(Children children);

    std::uint32_t arity() const noexcept;
    const Children& children() const noexcept override;

private:
    Children children_;
};

NodePtr operator+(NodePtr lhs, NodePtr rhs);
NodePtr operator+(NodePtr lhs, double rhs);
NodePtr operator+(double lhs, NodePtr rhs);
NodePtr operator-(NodePtr lhs, NodePtr rhs);
NodePtr operator-(NodePtr lhs, double rhs);
NodePtr operator-(double lhs, NodePtr rhs);
NodePtr operator*(NodePtr lhs, NodePtr rhs);
NodePtr operator*(NodePtr lhs, double rhs);
NodePtr operator*(double lhs, NodePtr rhs);
NodePtr operator+(NodePtr value);
NodePtr operator-(NodePtr value);

}  // namespace albert
