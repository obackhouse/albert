#include "ast.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace albert {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_byte(std::uint64_t& hash, std::uint8_t byte) noexcept {
    hash ^= byte;
    hash *= kFnvPrime;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        hash_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void hash_string(std::uint64_t& hash, const std::string& value) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const auto byte : value) {
        hash_byte(hash, static_cast<std::uint8_t>(byte));
    }
}

std::uint64_t hash_node(const Node& node) noexcept {
    std::uint64_t hash = kFnvOffset;
    hash_byte(hash, static_cast<std::uint8_t>(node.kind()));
    if (const auto* scalar = dynamic_cast<const Scalar*>(&node)) {
        static_assert(sizeof(double) == sizeof(std::uint64_t),
                      "AST hashes require 64-bit doubles");
        std::uint64_t bits = 0;
        const auto value = scalar->value();
        if (value != 0.0) {
            std::memcpy(&bits, &value, sizeof(bits));
        }
        hash_u64(hash, bits);
        return hash;
    }
    if (const auto* tensor = dynamic_cast<const Tensor*>(&node)) {
        hash_string(hash, tensor->symbol());
        hash_u64(hash, tensor->indices().size());
        for (const auto& index : tensor->indices()) {
            hash_string(hash, index);
        }
        return hash;
    }
    hash_u64(hash, node.children().size());
    for (const auto& child : node.children()) {
        hash_u64(hash, child->structural_hash());
    }
    return hash;
}

bool scalar_values_equal(double lhs, double rhs) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "AST equality requires 64-bit doubles");
    std::uint64_t lhs_bits = 0;
    std::uint64_t rhs_bits = 0;
    if (lhs != 0.0) {
        std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
    }
    if (rhs != 0.0) {
        std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
    }
    return lhs_bits == rhs_bits;
}

}  // namespace

CanonicalKey::CanonicalKey(NodePtr node) : node_(std::move(node)) {}

std::uint64_t CanonicalKey::hash() const noexcept {
    return node_->structural_hash();
}

bool CanonicalKey::operator==(const CanonicalKey& other) const noexcept {
    return node_->structurally_equal(*other.node_);
}

CanonicalKey canonical_key(const NodePtr& node) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
    return CanonicalKey(node);
}

std::uint64_t Node::structural_hash() const noexcept {
    if (!hash_cached_.load(std::memory_order_acquire)) {
        const auto value = hash_node(*this);
        structural_hash_.store(value, std::memory_order_relaxed);
        hash_cached_.store(true, std::memory_order_release);
    }
    return structural_hash_.load(std::memory_order_relaxed);
}

bool Node::structurally_equal(const Node& other) const noexcept {
    if (this == &other) {
        return true;
    }
    if (kind() != other.kind() || structural_hash() != other.structural_hash()) {
        return false;
    }
    if (const auto* lhs = dynamic_cast<const Scalar*>(this)) {
        const auto* rhs = dynamic_cast<const Scalar*>(&other);
        return rhs && scalar_values_equal(lhs->value(), rhs->value());
    }
    if (const auto* lhs = dynamic_cast<const Tensor*>(this)) {
        const auto* rhs = dynamic_cast<const Tensor*>(&other);
        return rhs && lhs->symbol() == rhs->symbol() &&
               lhs->indices() == rhs->indices();
    }
    const auto& lhs_children = children();
    const auto& rhs_children = other.children();
    if (lhs_children.size() != rhs_children.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs_children.size(); ++i) {
        if (!lhs_children[i]->structurally_equal(*rhs_children[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace albert
