#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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

std::string canonical_key(const NodePtr& node);

class Node {
public:
    virtual ~Node() = default;
    NodeKind kind() const noexcept;
    virtual const Children& children() const noexcept;

protected:
    explicit Node(NodeKind kind) noexcept;

private:
    NodeKind kind_;
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
