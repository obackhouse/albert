#include "expression.h"

#include <stdexcept>
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
}

const std::shared_ptr<Tensor>& Expression::lhs() const noexcept {
    return lhs_;
}

const NodePtr& Expression::rhs() const noexcept {
    return rhs_;
}

}  // namespace albert
