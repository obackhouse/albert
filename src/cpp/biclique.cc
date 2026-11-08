#include "biclique.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

namespace albert {
namespace {

using VertexSet = std::set<std::size_t>;
using Adjacency = std::vector<VertexSet>;

std::vector<std::string> sorted_unique(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<std::string> set_intersection(const std::vector<std::string>& left,
                                         const std::vector<std::string>& right) {
    std::vector<std::string> result;
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::back_inserter(result));
    return result;
}

std::vector<std::string> set_union(const std::vector<std::string>& left,
                                   const std::vector<std::string>& right) {
    std::vector<std::string> result;
    std::set_union(left.begin(), left.end(), right.begin(), right.end(),
                   std::back_inserter(result));
    return result;
}

std::vector<std::string> set_difference(const std::vector<std::string>& left,
                                        const std::vector<std::string>& right) {
    std::vector<std::string> result;
    std::set_difference(left.begin(), left.end(), right.begin(), right.end(),
                        std::back_inserter(result));
    return result;
}

bool close_like_python(double left, double right) {
    if (left == right) {
        return true;
    }
    if (!std::isfinite(left) || !std::isfinite(right)) {
        return false;
    }
    constexpr double relative_tolerance = 1e-9;
    return std::abs(left - right) <=
           relative_tolerance * std::max(std::abs(left), std::abs(right));
}

VertexSet intersection(const VertexSet& left, const VertexSet& right) {
    VertexSet result;
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::inserter(result, result.end()));
    return result;
}

std::size_t intersection_size(const VertexSet& left, const VertexSet& right) {
    std::size_t result = 0;
    auto l = left.begin();
    auto r = right.begin();
    while (l != left.end() && r != right.end()) {
        if (*l == *r) {
            ++result;
            ++l;
            ++r;
        } else if (*l < *r) {
            ++l;
        } else {
            ++r;
        }
    }
    return result;
}

void maximal_cliques(const Adjacency& adjacency, VertexSet current,
                     VertexSet possible, VertexSet excluded,
                     std::vector<VertexSet>& results) {
    if (possible.empty() && excluded.empty()) {
        results.push_back(std::move(current));
        return;
    }

    std::size_t pivot = adjacency.size();
    std::size_t best_neighbors = 0;
    VertexSet pivot_candidates = possible;
    pivot_candidates.insert(excluded.begin(), excluded.end());
    for (const auto candidate : pivot_candidates) {
        const auto neighbors = intersection_size(possible, adjacency[candidate]);
        if (pivot == adjacency.size() || neighbors > best_neighbors) {
            pivot = candidate;
            best_neighbors = neighbors;
        }
    }

    VertexSet candidates;
    for (const auto vertex : possible) {
        if (pivot == adjacency.size() || adjacency[pivot].find(vertex) == adjacency[pivot].end()) {
            candidates.insert(vertex);
        }
    }

    for (const auto vertex : candidates) {
        auto next_current = current;
        next_current.insert(vertex);
        maximal_cliques(adjacency, std::move(next_current),
                        intersection(possible, adjacency[vertex]),
                        intersection(excluded, adjacency[vertex]), results);
        possible.erase(vertex);
        excluded.insert(vertex);
    }
}

bool duplicate_terms(const BipartiteGraph& graph,
                     const std::vector<std::size_t>& left,
                     const std::vector<std::size_t>& right) {
    std::vector<NodePtr> terms;
    for (const auto l : left) {
        for (const auto r : right) {
            const auto* edge = graph.edge(l, r);
            if (edge) {
                if (std::any_of(terms.begin(), terms.end(), [&](const NodePtr& term) {
                        return term->structurally_equal(*edge->term);
                    })) {
                    return true;
                }
                terms.push_back(edge->term);
            }
        }
    }
    return false;
}

Adjacency extended_adjacency(const BipartiteGraph& graph) {
    const auto left_count = graph.left_vertices().size();
    const auto right_count = graph.right_vertices().size();
    Adjacency adjacency(left_count + right_count);
    const auto add_undirected = [&](std::size_t left, std::size_t right) {
        adjacency[left].insert(right);
        adjacency[right].insert(left);
    };

    // Albert completes each partition to a clique before calling NetworkX's
    // pivoted maximal-clique search. Each mixed maximal clique is a biclique.
    for (std::size_t left = 0; left < left_count; ++left) {
        for (std::size_t other = left + 1; other < left_count; ++other) {
            add_undirected(left, other);
        }
    }
    for (std::size_t right = 0; right < right_count; ++right) {
        for (std::size_t other = right + 1; other < right_count; ++other) {
            add_undirected(left_count + right, left_count + other);
        }
    }
    for (const auto& [left, right] : graph.edges()) {
        add_undirected(left, left_count + right);
    }
    return adjacency;
}

}  // namespace

std::vector<std::string> BicliqueVertex::indices() const {
    std::vector<std::string> result;
    std::set<std::string> seen;
    for (const auto& tensor : tensors) {
        if (!tensor) {
            throw std::invalid_argument("biclique vertices cannot contain null tensors");
        }
        for (const auto& index : tensor->indices()) {
            if (seen.insert(index).second) {
                result.push_back(index);
            }
        }
    }
    return result;
}

bool BicliqueVertex::operator==(const BicliqueVertex& other) const noexcept {
    if (tensors.size() != other.tensors.size()) {
        return false;
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i] || !other.tensors[i] ||
            !tensors[i]->structurally_equal(*other.tensors[i])) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> IndexPartition::indices_left() const {
    return set_union(sorted_unique(external_left), sorted_unique(internal));
}

std::vector<std::string> IndexPartition::indices_right() const {
    return set_union(sorted_unique(external_right), sorted_unique(internal));
}

std::vector<std::string> IndexPartition::output() const {
    return set_union(sorted_unique(external_left), sorted_unique(external_right));
}

bool IndexPartition::operator==(const IndexPartition& other) const {
    return external_left == other.external_left &&
           external_right == other.external_right && internal == other.internal;
}

IndexPartition IndexPartition::from_vertices(const Mul& expression,
                                             const BicliqueVertex& left,
                                             const BicliqueVertex& right) {
    auto expression_external = sorted_unique(expression.external_indices());
    const auto left_indices = sorted_unique(left.indices());
    const auto right_indices = sorted_unique(right.indices());
    const auto common = set_intersection(left_indices, right_indices);
    return {
        set_intersection(left_indices, expression_external),
        set_intersection(right_indices, expression_external),
        common,
    };
}

std::size_t IndexPartitionHash::operator()(const IndexPartition& partition) const noexcept {
    std::size_t hash = 0xcbf29ce484222325ULL;
    const auto combine = [&hash](const std::vector<std::string>& indices,
                                 std::size_t marker) {
        hash ^= marker;
        hash *= 0x100000001b3ULL;
        for (const auto& index : indices) {
            hash ^= std::hash<std::string>{}(index);
            hash *= 0x100000001b3ULL;
        }
    };
    combine(partition.external_left, 1);
    combine(partition.external_right, 2);
    combine(partition.internal, 3);
    return hash;
}

std::size_t BipartiteGraph::add_vertex(std::vector<BicliqueVertex>& vertices,
                                      BicliqueVertex vertex) {
    const auto found = std::find(vertices.begin(), vertices.end(), vertex);
    if (found != vertices.end()) {
        return static_cast<std::size_t>(found - vertices.begin());
    }
    vertices.push_back(std::move(vertex));
    return vertices.size() - 1;
}

std::pair<std::size_t, std::size_t> BipartiteGraph::add_edge(
    BicliqueVertex left, BicliqueVertex right, NodePtr term,
    double coefficient, double excess) {
    if (!term) {
        throw std::invalid_argument("biclique edges require a term");
    }
    const auto left_index = add_vertex(left_vertices_, std::move(left));
    const auto right_index = add_vertex(right_vertices_, std::move(right));
    if (adjacency_.size() < left_vertices_.size()) {
        adjacency_.resize(left_vertices_.size());
    }
    auto& row = adjacency_[left_index];
    const auto found = row.find(right_index);
    if (found != row.end()) {
        found->second.excess = std::min(found->second.excess, excess);
        return {left_index, right_index};
    }
    row.emplace(right_index, BicliqueEdge{excess, std::move(term), coefficient});
    edges_.emplace_back(left_index, right_index);
    return {left_index, right_index};
}

const std::vector<BicliqueVertex>& BipartiteGraph::left_vertices() const noexcept {
    return left_vertices_;
}

const std::vector<BicliqueVertex>& BipartiteGraph::right_vertices() const noexcept {
    return right_vertices_;
}

const std::vector<std::pair<std::size_t, std::size_t>>& BipartiteGraph::edges() const noexcept {
    return edges_;
}

const BicliqueEdge* BipartiteGraph::edge(std::size_t left,
                                        std::size_t right) const noexcept {
    if (left >= adjacency_.size()) {
        return nullptr;
    }
    const auto found = adjacency_[left].find(right);
    return found == adjacency_[left].end() ? nullptr : &found->second;
}

bool BipartiteGraph::has_edge(std::size_t left, std::size_t right) const noexcept {
    return edge(left, right) != nullptr;
}

RankOneDecomposition decompose_edge_coefficients(
    const BipartiteGraph& graph, const std::vector<std::size_t>& left,
    const std::vector<std::size_t>& right) {
    RankOneDecomposition result;
    result.left_coefficients.assign(left.size(), 0.0);
    result.right_coefficients.assign(right.size(), 0.0);
    if (left.empty() || right.empty()) {
        return result;
    }

    for (const auto left_vertex : left) {
        for (const auto right_vertex : right) {
            if (!graph.has_edge(left_vertex, right_vertex)) {
                throw std::invalid_argument(
                    "coefficient decomposition requires a complete biclique");
            }
        }
    }

    std::size_t anchor_left = left.size();
    std::size_t anchor_right = right.size();
    for (std::size_t i = 0; i < left.size() && anchor_left == left.size(); ++i) {
        for (std::size_t j = 0; j < right.size(); ++j) {
            const auto* edge = graph.edge(left[i], right[j]);
            if (edge->coefficient != 0.0) {
                result.coefficient = edge->coefficient;
                anchor_left = i;
                anchor_right = j;
                break;
            }
        }
    }
    if (anchor_left == left.size()) {
        return result;
    }

    for (std::size_t i = 0; i < left.size(); ++i) {
        result.left_coefficients[i] =
            graph.edge(left[i], right[anchor_right])->coefficient / result.coefficient;
    }
    result.left_coefficients[anchor_left] = 1.0;
    for (std::size_t j = 0; j < right.size(); ++j) {
        result.right_coefficients[j] =
            graph.edge(left[anchor_left], right[j])->coefficient / result.coefficient;
    }
    result.right_coefficients[anchor_right] = 1.0;

    for (std::size_t i = 0; i < left.size(); ++i) {
        for (std::size_t j = 0; j < right.size(); ++j) {
            const auto expected = result.coefficient * result.left_coefficients[i] *
                                  result.right_coefficients[j];
            if (!close_like_python(graph.edge(left[i], right[j])->coefficient, expected)) {
                result.coefficient = 0.0;
                result.left_coefficients.assign(left.size(), 0.0);
                result.right_coefficients.assign(right.size(), 0.0);
                return result;
            }
        }
    }
    return result;
}

std::vector<ConstrictableBiclique> find_constrictable_bicliques(
    const BipartiteGraph& graph, const IndexPartition& partition,
    std::size_t expression_id) {
    std::vector<ConstrictableBiclique> result;
    const auto left_count = graph.left_vertices().size();
    const auto adjacency = extended_adjacency(graph);
    VertexSet possible;
    for (std::size_t vertex = 0; vertex < adjacency.size(); ++vertex) {
        possible.insert(vertex);
    }
    std::vector<VertexSet> cliques;
    maximal_cliques(adjacency, {}, std::move(possible), {}, cliques);

    for (const auto& clique : cliques) {
        std::vector<std::size_t> left;
        std::vector<std::size_t> right;
        for (const auto vertex : clique) {
            if (vertex < left_count) {
                left.push_back(vertex);
            } else {
                right.push_back(vertex - left_count);
            }
        }
        if (left.empty() || right.empty() || duplicate_terms(graph, left, right)) {
            continue;
        }

        std::size_t left_tensor_count = 0;
        for (const auto vertex : left) {
            left_tensor_count += graph.left_vertices()[vertex].tensors.size();
        }
        std::size_t right_tensor_count = 0;
        for (const auto vertex : right) {
            right_tensor_count += graph.right_vertices()[vertex].tensors.size();
        }
        if (left_tensor_count == 1 && right_tensor_count == 1) {
            continue;
        }

        const auto decomposition = decompose_edge_coefficients(graph, left, right);
        if (decomposition.coefficient == 0.0) {
            continue;
        }
        result.push_back({
            {left, decomposition.left_coefficients, sorted_unique(partition.external_left)},
            {right, decomposition.right_coefficients, sorted_unique(partition.external_right)},
            decomposition.coefficient,
            sorted_unique(partition.internal),
            expression_id,
        });
    }
    return result;
}

}  // namespace albert
