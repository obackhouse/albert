#pragma once

#include "algebra.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace albert {

struct BicliqueVertex {
    std::vector<std::shared_ptr<Tensor>> tensors;

    std::vector<std::string> indices() const;
    bool operator==(const BicliqueVertex& other) const noexcept;
};

struct IndexPartition {
    std::vector<std::string> external_left;
    std::vector<std::string> external_right;
    std::vector<std::string> internal;

    std::vector<std::string> indices_left() const;
    std::vector<std::string> indices_right() const;
    std::vector<std::string> output() const;
    bool operator==(const IndexPartition& other) const;

    static IndexPartition from_vertices(const Mul& expression,
                                       const BicliqueVertex& left,
                                       const BicliqueVertex& right);
};

struct IndexPartitionHash {
    std::size_t operator()(const IndexPartition& partition) const noexcept;
};

struct BicliqueEdge {
    double excess = 0.0;
    NodePtr term;
    double coefficient = 1.0;
};

class BipartiteGraph {
public:
    std::pair<std::size_t, std::size_t> add_edge(
        BicliqueVertex left, BicliqueVertex right, NodePtr term,
        double coefficient, double excess);

    const std::vector<BicliqueVertex>& left_vertices() const noexcept;
    const std::vector<BicliqueVertex>& right_vertices() const noexcept;
    const std::vector<std::pair<std::size_t, std::size_t>>& edges() const noexcept;
    const BicliqueEdge* edge(std::size_t left, std::size_t right) const noexcept;
    bool has_edge(std::size_t left, std::size_t right) const noexcept;

private:
    std::size_t add_vertex(std::vector<BicliqueVertex>& vertices,
                           BicliqueVertex vertex);

    std::vector<BicliqueVertex> left_vertices_;
    std::vector<BicliqueVertex> right_vertices_;
    std::vector<std::pair<std::size_t, std::size_t>> edges_;
    std::vector<std::unordered_map<std::size_t, BicliqueEdge>> adjacency_;
};

struct RankOneDecomposition {
    double coefficient = 0.0;
    std::vector<double> left_coefficients;
    std::vector<double> right_coefficients;
};

struct BicliqueSubset {
    std::vector<std::size_t> vertices;
    std::vector<double> coefficients;
    std::vector<std::string> external_indices;
};

struct ConstrictableBiclique {
    BicliqueSubset left;
    BicliqueSubset right;
    double coefficient = 0.0;
    std::vector<std::string> internal_indices;
    std::size_t expression_id = 0;
};

RankOneDecomposition decompose_edge_coefficients(
    const BipartiteGraph& graph, const std::vector<std::size_t>& left,
    const std::vector<std::size_t>& right);

std::vector<ConstrictableBiclique> find_constrictable_bicliques(
    const BipartiteGraph& graph, const IndexPartition& partition,
    std::size_t expression_id = 0);

}  // namespace albert
