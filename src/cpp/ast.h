#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace albert {

enum class NodeKind : std::uint8_t {
    Scalar,
    Tensor,
    Add,
    Mul,
};

class Node;
using NodePtr = std::shared_ptr<Node>;
using Children = std::vector<NodePtr>;

class CanonicalKey {
public:
    std::uint64_t hash() const noexcept;
    bool operator==(const CanonicalKey& other) const noexcept;

private:
    friend CanonicalKey canonical_key(const NodePtr& node);

    explicit CanonicalKey(NodePtr node);
    NodePtr node_;
};

CanonicalKey canonical_key(const NodePtr& node);
NodePtr map_indices(
    const NodePtr& node,
    const std::unordered_map<std::string, std::string>& mapping);

class Node {
public:
    virtual ~Node() = default;
    NodeKind kind() const noexcept;
    virtual const Children& children() const noexcept;
    std::vector<std::string> external_indices() const;
    std::vector<std::string> internal_indices() const;
    std::uint64_t structural_hash() const noexcept;
    bool structurally_equal(const Node& other) const noexcept;

protected:
    explicit Node(NodeKind kind) noexcept;

private:
    NodeKind kind_;
    mutable std::atomic<bool> hash_cached_{false};
    mutable std::atomic<std::uint64_t> structural_hash_{0};
};

class Scalar final : public Node {
public:
    explicit Scalar(double value);
    double value() const noexcept;

private:
    double value_;
};

class Tensor final : public Node {
public:
    Tensor(std::string symbol, std::vector<std::string> indices);

    const std::string& symbol() const noexcept;
    std::uint32_t rank() const noexcept;
    const std::vector<std::string>& indices() const noexcept;

private:
    std::string symbol_;
    std::vector<std::string> indices_;
};

}  // namespace albert
