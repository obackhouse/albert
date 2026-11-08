#pragma once

#include "ast.h"

#include <memory>

namespace albert {

class Expression {
public:
    Expression(std::shared_ptr<Tensor> lhs, NodePtr rhs);

    const std::shared_ptr<Tensor>& lhs() const noexcept;
    const NodePtr& rhs() const noexcept;

private:
    std::shared_ptr<Tensor> lhs_;
    NodePtr rhs_;
};

}  // namespace albert
