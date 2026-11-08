#include "simplify.h"

#include "algebra.h"
#include "search.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace albert {
namespace {

using Monomial = Children;
using Polynomial = std::vector<Monomial>;
constexpr std::uint64_t kFactorHashOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFactorHashPrime = 1099511628211ULL;

void hash_factor_byte(std::uint64_t& hash, std::uint8_t byte) noexcept {
    hash ^= byte;
    hash *= kFactorHashPrime;
}

std::uint64_t factors_hash(const Children& factors) noexcept {
    std::uint64_t hash = kFactorHashOffset;
    const auto size = static_cast<std::uint64_t>(factors.size());
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        hash_factor_byte(hash, static_cast<std::uint8_t>(size >> shift));
    }
    for (const auto& factor : factors) {
        const auto value = factor->structural_hash();
        for (unsigned int shift = 0; shift < 64; shift += 8) {
            hash_factor_byte(hash, static_cast<std::uint8_t>(value >> shift));
        }
    }
    return hash;
}

bool same_factors(const Children& lhs, const Children& rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!lhs[i]->structurally_equal(*rhs[i])) {
            return false;
        }
    }
    return true;
}

void require_node(const NodePtr& node) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
}

Polynomial polynomial(const NodePtr& node) {
    if (node->kind() == NodeKind::Scalar || node->kind() == NodeKind::Tensor) {
        return {{node}};
    }
    if (node->kind() == NodeKind::Add) {
        Polynomial result;
        for (const auto& child : node->children()) {
            auto terms = polynomial(child);
            result.insert(result.end(), std::make_move_iterator(terms.begin()),
                          std::make_move_iterator(terms.end()));
        }
        return result;
    }

    Polynomial result(1);
    for (const auto& child : node->children()) {
        const auto child_terms = polynomial(child);
        if (child_terms.empty()) {
            return {};
        }
        if (result.size() > std::numeric_limits<std::size_t>::max() /
                                child_terms.size()) {
            throw std::length_error("expanded AST is too large");
        }
        Polynomial product;
        product.reserve(result.size() * child_terms.size());
        for (const auto& left : result) {
            for (const auto& right : child_terms) {
                Monomial factors;
                factors.reserve(left.size() + right.size());
                factors.insert(factors.end(), left.begin(), left.end());
                factors.insert(factors.end(), right.begin(), right.end());
                product.push_back(std::move(factors));
            }
        }
        result = std::move(product);
    }
    return result;
}

NodePtr make_expanded(const Polynomial& terms) {
    Children summands;
    summands.reserve(terms.size());
    for (const auto& term : terms) {
        summands.push_back(std::make_shared<Mul>(term));
    }
    return std::make_shared<Add>(std::move(summands));
}

struct LikeTerm {
    Children factors;
    double coefficient;
};

LikeTerm split_coefficient(const NodePtr& term) {
    LikeTerm result{{}, 1.0};
    const auto* product = dynamic_cast<const Mul*>(term.get());
    if (!product) {
        if (const auto* scalar = dynamic_cast<const Scalar*>(term.get())) {
            result.coefficient = scalar->value();
        } else {
            result.factors.push_back(term);
        }
        return result;
    }
    for (const auto& factor : product->children()) {
        if (const auto* scalar = dynamic_cast<const Scalar*>(factor.get())) {
            result.coefficient *= scalar->value();
        } else {
            result.factors.push_back(factor);
        }
    }
    return result;
}

NodePtr make_like_term(const LikeTerm& term) {
    if (term.factors.empty()) {
        return std::make_shared<Scalar>(term.coefficient);
    }
    if (term.coefficient == 1.0) {
        if (term.factors.size() == 1) {
            return term.factors.front();
        }
        return std::make_shared<Mul>(term.factors);
    }
    Children factors;
    factors.reserve(term.factors.size() + 1);
    factors.push_back(std::make_shared<Scalar>(term.coefficient));
    factors.insert(factors.end(), term.factors.begin(), term.factors.end());
    return std::make_shared<Mul>(std::move(factors));
}

bool is_scalar_value(const NodePtr& node, double value) {
    const auto* scalar = dynamic_cast<const Scalar*>(node.get());
    return scalar && scalar->value() == value;
}

constexpr std::size_t kMaxExpansionTerms = 256;

std::size_t expansion_size(const NodePtr& node, std::size_t limit) {
    if (node->kind() == NodeKind::Scalar || node->kind() == NodeKind::Tensor) {
        return 1;
    }
    if (node->kind() == NodeKind::Add) {
        std::size_t total = 0;
        for (const auto& child : node->children()) {
            const auto size = expansion_size(child, limit);
            if (size > limit - total) {
                return limit + 1;
            }
            total += size;
        }
        return total;
    }
    std::size_t total = 1;
    for (const auto& child : node->children()) {
        const auto size = expansion_size(child, limit);
        if (size == 0) {
            return 0;
        }
        if (size > limit || total > limit / size) {
            return limit + 1;
        }
        total *= size;
    }
    return total;
}

bool is_fully_expanded(const NodePtr& node) {
    if (node->kind() == NodeKind::Add) {
        for (const auto& term : node->children()) {
            if (term->kind() != NodeKind::Mul) {
                return false;
            }
            for (const auto& factor : term->children()) {
                if (factor->kind() != NodeKind::Scalar &&
                    factor->kind() != NodeKind::Tensor) {
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

bool contains_add(const NodePtr& node) {
    if (node->kind() == NodeKind::Add) {
        return true;
    }
    for (const auto& child : node->children()) {
        if (contains_add(child)) {
            return true;
        }
    }
    return false;
}

}  // namespace

NodePtr expand(const NodePtr& node) {
    require_node(node);
    if (is_fully_expanded(node)) {
        return node;
    }
    return make_expanded(polynomial(node));
}

NodePtr collect(const NodePtr& node) {
    require_node(node);
    const auto* sum = dynamic_cast<const Add*>(node.get());
    if (!sum) {
        return node;
    }

    std::vector<LikeTerm> terms;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> term_indices;
    for (const auto& child : sum->children()) {
        auto candidate = split_coefficient(child);
        auto& bucket = term_indices[factors_hash(candidate.factors)];
        const auto match = std::find_if(bucket.begin(), bucket.end(), [&](std::size_t i) {
            return same_factors(terms[i].factors, candidate.factors);
        });
        if (match == bucket.end()) {
            bucket.push_back(terms.size());
            terms.push_back(std::move(candidate));
        } else {
            terms[*match].coefficient += candidate.coefficient;
        }
    }

    if (terms.size() == sum->children().size()) {
        bool unchanged = true;
        for (std::size_t i = 0; i < terms.size(); ++i) {
            if (terms[i].coefficient != 1.0 || terms[i].factors.size() != 1 ||
                terms[i].factors.front() != sum->children()[i]) {
                unchanged = false;
                break;
            }
        }
        if (unchanged) {
            return node;
        }
    }

    Children collected;
    collected.reserve(terms.size());
    for (const auto& term : terms) {
        collected.push_back(make_like_term(term));
    }
    return std::make_shared<Add>(std::move(collected));
}

NodePtr squeeze(const NodePtr& node) {
    require_node(node);
    if ((node->kind() == NodeKind::Add || node->kind() == NodeKind::Mul) &&
        node->children().size() == 1) {
        return node->children().front();
    }
    return node;
}

NodePtr remove_neutral_elements(const NodePtr& node) {
    require_node(node);
    if (node->kind() == NodeKind::Add) {
        Children children;
        for (const auto& child : node->children()) {
            if (!is_scalar_value(child, 0.0)) {
                children.push_back(child);
            }
        }
        if (children.size() == node->children().size()) {
            return node;
        }
        if (children.empty()) {
            return std::make_shared<Scalar>(0.0);
        }
        return std::make_shared<Add>(std::move(children));
    }
    if (node->kind() == NodeKind::Mul) {
        Children children;
        for (const auto& child : node->children()) {
            if (!is_scalar_value(child, 1.0)) {
                children.push_back(child);
            }
        }
        if (children.size() == node->children().size()) {
            return node;
        }
        if (children.empty()) {
            return std::make_shared<Scalar>(1.0);
        }
        return std::make_shared<Mul>(std::move(children));
    }
    return node;
}

NodePtr simplify(const NodePtr& node) {
    require_node(node);
    auto current = node;
    for (std::size_t pass = 0; pass < 8; ++pass) {
        auto updated = albert::apply(current, [](const NodePtr& current_node) {
            if (current_node->kind() == NodeKind::Add) {
                return collect(current_node);
            }
            return current_node;
        });
        updated = albert::apply(updated, [](const NodePtr& current_node) {
            return remove_neutral_elements(current_node);
        });
        updated = albert::apply(updated, [](const NodePtr& current_node) {
            return squeeze(current_node);
        });
        updated = albert::apply(updated, [](const NodePtr& current_node) {
            if (current_node->kind() == NodeKind::Mul &&
                contains_add(current_node) &&
                expansion_size(current_node, kMaxExpansionTerms) <=
                    kMaxExpansionTerms) {
                return expand(current_node);
            }
            return current_node;
        });
        if (current->structurally_equal(*updated)) {
            return updated;
        }
        current = std::move(updated);
    }
    return current;
}

}  // namespace albert
