#include "cse.h"

#include "algebra.h"
#include "search.h"
#include "simplify.h"
#include "sort.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace albert {
namespace {

struct GraphRecord {
    IndexPartition partition;
    BipartiteGraph graph;
};

struct ScoredBiclique {
    std::size_t graph_index = 0;
    ConstrictableBiclique biclique;
    double score = 0.0;
};

struct CachedPaths {
    NodePtr product;
    std::vector<ParenthesizationCandidate> candidates;
};

using ParenthesizationCache =
    std::unordered_map<std::uint64_t, std::vector<CachedPaths>>;

thread_local std::size_t cse_depth = 0;

class CSEDepthGuard {
public:
    CSEDepthGuard() { ++cse_depth; }
    ~CSEDepthGuard() { --cse_depth; }
};

std::vector<std::shared_ptr<Tensor>> tensors_in(const NodePtr& node) {
    std::vector<std::shared_ptr<Tensor>> tensors;
    for (const auto& found : search(node, [](const NodePtr& item) {
             return item->kind() == NodeKind::Tensor;
         })) {
        tensors.push_back(std::dynamic_pointer_cast<Tensor>(found));
    }
    return tensors;
}

bool has_three_use_index(const NodePtr& node) {
    std::unordered_map<std::string, std::size_t> occurrences;
    for (const auto& tensor : tensors_in(node)) {
        for (const auto& index : tensor->indices()) {
            if (++occurrences[index] > 2) {
                return true;
            }
        }
    }
    return false;
}

double scalar_coefficient(const NodePtr& node) {
    double coefficient = 1.0;
    for (const auto& found : search(node, [](const NodePtr& item) {
             return item->kind() == NodeKind::Scalar;
         })) {
        coefficient *= std::dynamic_pointer_cast<Scalar>(found)->value();
    }
    return coefficient;
}

std::vector<NodePtr> terms(const NodePtr& rhs) {
    const auto expanded = expand(rhs);
    if (const auto* add = dynamic_cast<const Add*>(expanded.get())) {
        return add->children();
    }
    return {expanded};
}

NodePtr parenthesize_products(const NodePtr& node, const IndexSpaces& spaces,
                             const SpaceSizes& sizes, const CostFactors& factors,
                             const CSEOptions& options,
                             ParenthesizationCache& path_cache) {
    if (node->kind() == NodeKind::Mul && has_three_use_index(node)) {
        return node;
    }
    if (node->kind() == NodeKind::Add) {
        Children children;
        children.reserve(node->children().size());
        bool changed = false;
        for (const auto& child : node->children()) {
            auto rewritten = parenthesize_products(child, spaces, sizes, factors, options,
                                                   path_cache);
            changed = changed || rewritten != child;
            children.push_back(std::move(rewritten));
        }
        return changed ? std::make_shared<Add>(std::move(children)) : node;
    }
    if (node->kind() != NodeKind::Mul) {
        return node;
    }

    Children children;
    children.reserve(node->children().size());
    bool changed = false;
    for (const auto& child : node->children()) {
        auto rewritten = parenthesize_products(child, spaces, sizes, factors, options,
                                               path_cache);
        changed = changed || rewritten != child;
        children.push_back(std::move(rewritten));
    }
    auto product = changed ? std::make_shared<Mul>(children)
                           : std::dynamic_pointer_cast<Mul>(node);
    if (!product) {
        return node;
    }

    Children tensors;
    double coefficient = 1.0;
    bool has_scalar = false;
    for (const auto& child : product->children()) {
        if (const auto* scalar = dynamic_cast<const Scalar*>(child.get())) {
            coefficient *= scalar->value();
            has_scalar = true;
        } else if (child->kind() == NodeKind::Tensor) {
            tensors.push_back(child);
        } else {
            return product;
        }
    }
    if (tensors.size() < 2) {
        return product;
    }
    Mul tensor_product(std::move(tensors));
    std::vector<ParenthesizationCandidate> candidates;
    auto& bucket = path_cache[tensor_product.structural_hash()];
    const auto cached = std::find_if(bucket.begin(), bucket.end(), [&](const auto& entry) {
        return entry.product->structurally_equal(tensor_product);
    });
    if (cached != bucket.end()) {
        candidates = cached->candidates;
    } else {
        PathRequest request;
        if (options.custom_path_requests) {
            request = options.path_requests.for_size(tensor_product.children().size());
        } else {
            request.dynamic = tensor_product.children().size() > options.exhaustive_limit;
            request.exhaustive = request.dynamic ? 0 : 1;
        }
        candidates = parenthesization_candidates(
            tensor_product, spaces, sizes, options.scaling_limits,
            request, 1, options.cost_model, factors);
        bucket.push_back({
            std::make_shared<Mul>(Children(tensor_product.children().begin(),
                                           tensor_product.children().end())),
            candidates});
    }
    if (candidates.empty()) {
        if (!options.scaling_limits.time.empty() || !options.scaling_limits.memory.empty()) {
            throw std::invalid_argument(
                "no parenthesization satisfies the scaling limits for a CSE input term");
        }
        return product;
    }
    auto result = apply_parenthesization(tensor_product, candidates.front());
    if (has_scalar && coefficient != 1.0) {
        result = std::make_shared<Mul>(Children{result, std::make_shared<Scalar>(coefficient)});
    }
    return result;
}

std::pair<BicliqueVertex, BicliqueVertex> final_vertices(
    const std::vector<std::shared_ptr<Tensor>>& tensors,
    const ParenthesizationCandidate& candidate) {
    std::vector<std::vector<std::size_t>> groups;
    groups.reserve(tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        groups.push_back({i});
    }
    if (candidate.path.empty()) {
        return {};
    }
    for (std::size_t p = 0; p + 1 < candidate.path.size(); ++p) {
        auto [i, j] = candidate.path[p];
        if (i > j) {
            std::swap(i, j);
        }
        if (j >= groups.size()) {
            return {};
        }
        auto merged = groups[i];
        merged.insert(merged.end(), groups[j].begin(), groups[j].end());
        groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(j));
        groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(i));
        groups.push_back(std::move(merged));
    }
    auto [i, j] = candidate.path.back();
    if (i > j) {
        std::swap(i, j);
    }
    if (j >= groups.size()) {
        return {};
    }
    BicliqueVertex left;
    BicliqueVertex right;
    for (const auto index : groups[i]) {
        left.tensors.push_back(tensors[index]);
    }
    for (const auto index : groups[j]) {
        right.tensors.push_back(tensors[index]);
    }
    const auto less = [](const BicliqueVertex& a, const BicliqueVertex& b) {
        for (std::size_t i = 0; i < std::min(a.tensors.size(), b.tensors.size()); ++i) {
            if (a.tensors[i]->symbol() != b.tensors[i]->symbol()) {
                return a.tensors[i]->symbol() < b.tensors[i]->symbol();
            }
            if (a.tensors[i]->indices() != b.tensors[i]->indices()) {
                return a.tensors[i]->indices() < b.tensors[i]->indices();
            }
        }
        return a.tensors.size() < b.tensors.size();
    };
    if (less(right, left)) {
        std::swap(left, right);
    }
    return {std::move(left), std::move(right)};
}

std::vector<GraphRecord> build_graphs(const std::vector<Expression>& expressions,
                                      const IndexSpaces& spaces,
                                      const SpaceSizes& sizes,
                                      const CSEOptions& options,
                                      ParenthesizationCache& path_cache) {
    std::vector<GraphRecord> graphs;
    for (std::size_t expression_id = 0; expression_id < expressions.size(); ++expression_id) {
        std::unordered_map<IndexPartition, BipartiteGraph, IndexPartitionHash> expression_graphs;
        std::vector<std::pair<NodePtr, std::size_t>> unique_terms;
        for (const auto& term : terms(expressions[expression_id].rhs())) {
            const auto found = std::find_if(
                unique_terms.begin(), unique_terms.end(), [&](const auto& item) {
                    return item.first->structurally_equal(*term);
                });
            if (found == unique_terms.end()) {
                unique_terms.emplace_back(term, 1);
            } else {
                ++found->second;
            }
        }
        for (const auto& [term, multiplicity] : unique_terms) {
            if (has_three_use_index(term)) {
                continue;
            }
            const auto term_tensors = tensors_in(term);
            if (term_tensors.size() < 2) {
                continue;
            }
            auto product = std::make_shared<Mul>(Children(term_tensors.begin(), term_tensors.end()));
            auto& bucket = path_cache[product->structural_hash()];
            auto cached = std::find_if(bucket.begin(), bucket.end(), [&](const auto& entry) {
                return entry.product->structurally_equal(*product);
            });
            if (cached == bucket.end()) {
                CostModel path_flop_model;
                path_flop_model.add = 0.0;
                path_flop_model.read = 0.0;
                path_flop_model.write = 0.0;
                std::vector<ParenthesizationCandidate> candidates;
                const auto& request = options.path_requests.for_size(term_tensors.size());
                if (request.dynamic && request.exhaustive > 0 &&
                    options.max_path_samples > 0 &&
                    options.scaling_limits.time.empty() &&
                    options.scaling_limits.memory.empty()) {
                    // Preserve the default sampler: seed with the FLOP-optimal
                    // DP path, then rank exhaustive samples by FLOPs + 64 * writes.
                    candidates = parenthesization_candidates(
                        *product, spaces, sizes, options.scaling_limits,
                        PathRequest{0, 0, 0, true}, 1, path_flop_model);
                    const auto exhaustive_count = std::min(
                        request.exhaustive, options.max_path_samples > 0
                                                ? options.max_path_samples - 1
                                                : std::size_t{0});
                    auto sample_model = path_flop_model;
                    sample_model.write = 64.0;
                    auto sampled = parenthesization_candidates(
                        *product, spaces, sizes, options.scaling_limits,
                        PathRequest{exhaustive_count, 0, 0, false},
                        exhaustive_count, sample_model);
                    for (auto& candidate : sampled) {
                        if (std::none_of(candidates.begin(), candidates.end(),
                                         [&](const auto& existing) {
                                             return existing.path == candidate.path;
                                         })) {
                            candidates.push_back(std::move(candidate));
                        }
                    }
                } else {
                    candidates = parenthesization_candidates(
                        *product, spaces, sizes, options.scaling_limits,
                        request,
                        options.max_path_samples, path_flop_model);
                }
                bucket.push_back({product, std::move(candidates)});
                cached = std::prev(bucket.end());
            }
            const auto& candidates = cached->candidates;
            if (candidates.empty()) {
                continue;
            }
            const auto best = std::min_element(candidates.begin(), candidates.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.cost.multiplications < rhs.cost.multiplications;
                });
            for (const auto& candidate : candidates) {
                const auto [left, right] = final_vertices(term_tensors, candidate);
                if (left.tensors.empty() || right.tensors.empty()) {
                    continue;
                }
                const auto partition = IndexPartition::from_vertices(*product, left, right);
                auto& graph = expression_graphs[partition];
                graph.add_edge(left, right, term,
                               scalar_coefficient(term) * static_cast<double>(multiplicity),
                               std::max(0.0, candidate.cost.multiplications -
                                                  best->cost.multiplications));
            }
        }
        for (auto& [partition, graph] : expression_graphs) {
            graphs.push_back({partition, std::move(graph)});
        }
    }
    std::stable_sort(graphs.begin(), graphs.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.partition.external_left, lhs.partition.external_right,
                        lhs.partition.internal) <
               std::tie(rhs.partition.external_left, rhs.partition.external_right,
                        rhs.partition.internal);
    });
    return graphs;
}

double index_product(const std::vector<std::string>& indices,
                     const IndexSpaces& spaces, const SpaceSizes& sizes) {
    std::unordered_map<std::string, std::string> owner;
    for (const auto& [space, labels] : spaces) {
        for (const auto& label : labels) {
            const auto inserted = owner.emplace(label, space);
            if (!inserted.second && inserted.first->second != space) {
                throw std::invalid_argument("an index cannot belong to multiple spaces");
            }
        }
    }
    double result = 1.0;
    std::unordered_set<std::string> unique;
    for (const auto& index : indices) {
        if (!unique.insert(index).second) {
            continue;
        }
        const auto space = owner.find(index);
        if (space == owner.end() || sizes.find(space->second) == sizes.end()) {
            throw std::invalid_argument("index " + index + " has no declared space size");
        }
        result *= static_cast<double>(sizes.at(space->second));
    }
    return result;
}

double gross_saving(const ConstrictableBiclique& cb, const IndexSpaces& spaces,
                    const SpaceSizes& sizes) {
    const double l = static_cast<double>(cb.left.vertices.size());
    const double r = static_cast<double>(cb.right.vertices.size());
    const double el = index_product(cb.left.external_indices, spaces, sizes);
    const double er = index_product(cb.right.external_indices, spaces, sizes);
    const double s = index_product(cb.internal_indices, spaces, sizes);
    return std::max(0.0, l * r * el * er * s - (l * el * s + r * er * s + el * er * s));
}

std::vector<ScoredBiclique> select_bicliques(const std::vector<GraphRecord>& graphs,
                                              const IndexSpaces& spaces,
                                              const SpaceSizes& sizes,
                                              const CostFactors& cost_factors,
                                              const CostModel& cost_model,
                                              std::size_t maximum) {
    std::vector<std::vector<ScoredBiclique>> graph_results(graphs.size());
#ifdef _OPENMP
    // Keep small scoring batches to two workers. For larger graph batches,
    // scale with the estimated pair work rather than graph count alone: a few
    // dense graphs can cost more than many tiny ones.
    std::size_t pair_work = 0;
    for (const auto& record : graphs) {
        const auto left = record.graph.left_vertices().size();
        const auto right = record.graph.right_vertices().size();
        const auto graph_work = right != 0 && left >
                std::numeric_limits<std::size_t>::max() / right
            ? std::numeric_limits<std::size_t>::max()
            : left * right;
        pair_work = graph_work > std::numeric_limits<std::size_t>::max() - pair_work
            ? std::numeric_limits<std::size_t>::max()
            : pair_work + graph_work;
    }
    const int scoring_threads = pair_work >= 16'384
        ? omp_get_max_threads()
        : std::min(2, omp_get_max_threads());
#pragma omp parallel for schedule(dynamic) num_threads(scoring_threads) if (graphs.size() >= 4)
#endif
    for (std::ptrdiff_t graph_index = 0;
         graph_index < static_cast<std::ptrdiff_t>(graphs.size()); ++graph_index) {
        const auto index = static_cast<std::size_t>(graph_index);
        const auto& record = graphs[index];
        auto& scored = graph_results[index];
        for (auto cb : find_constrictable_bicliques(record.graph, record.partition)) {
            if (cb.left.vertices.empty() || cb.right.vertices.empty()) {
                continue;
            }
            double excess = 0.0;
            double factor_sum = 0.0;
            std::size_t edge_count = 0;
            for (const auto left : cb.left.vertices) {
                for (const auto right : cb.right.vertices) {
                    if (const auto* edge = record.graph.edge(left, right)) {
                        excess += edge->excess;
                        if (cost_model.apply_cost_factors) {
                            ++edge_count;
                            std::unordered_set<std::string> symbols;
                            for (const auto& tensor : tensors_in(edge->term)) {
                                symbols.insert(tensor->symbol());
                            }
                            double factor = 1.0;
                            for (const auto& symbol : symbols) {
                                const auto found = cost_factors.find(symbol);
                                if (found == cost_factors.end()) {
                                    continue;
                                }
                                if (cost_model.cost_factor_aggregation ==
                                    CostFactorAggregation::Maximum) {
                                    factor = std::max(factor, static_cast<double>(found->second));
                                } else {
                                    factor *= static_cast<double>(found->second);
                                }
                            }
                            factor_sum += factor;
                        }
                    }
                }
            }
            const double average_factor = !cost_model.apply_cost_factors || edge_count == 0
                ? 1.0
                : factor_sum / static_cast<double>(edge_count);
            const double score = gross_saving(cb, spaces, sizes) * average_factor - excess;
            scored.push_back({index, std::move(cb), score});
        }
    }
    std::vector<ScoredBiclique> result;
    for (auto& scored : graph_results) {
        for (auto& candidate : scored) {
            result.push_back(std::move(candidate));
        }
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score > rhs.score;
    });
    if (result.size() > maximum) {
        result.resize(maximum);
    }
    return result;
}

std::string next_name(const std::string& format, std::size_t number) {
    const auto placeholder = format.find("{}");
    if (placeholder == std::string::npos) {
        return format + std::to_string(number);
    }
    return format.substr(0, placeholder) + std::to_string(number) +
           format.substr(placeholder + 2);
}

std::string next_available_name(const std::string& format,
                                const std::unordered_set<std::string>& reserved) {
    auto number = reserved.size();
    auto name = next_name(format, number);
    while (reserved.find(name) != reserved.end()) {
        name = next_name(format, ++number);
    }
    return name;
}

std::vector<NodePtr> covered_terms(const BipartiteGraph& graph,
                                   const ConstrictableBiclique& cb) {
    std::vector<NodePtr> covered;
    for (const auto left : cb.left.vertices) {
        for (const auto right : cb.right.vertices) {
            if (const auto* edge = graph.edge(left, right)) {
                if (std::none_of(covered.begin(), covered.end(), [&](const NodePtr& item) {
                        return item->structurally_equal(*edge->term);
                    })) {
                    covered.push_back(edge->term);
                }
            }
        }
    }
    return covered;
}

NodePtr build_side(const std::vector<BicliqueVertex>& vertices,
                   const BicliqueSubset& subset) {
    Children summands;
    summands.reserve(subset.vertices.size());
    for (std::size_t i = 0; i < subset.vertices.size(); ++i) {
        const auto vertex = subset.vertices[i];
        Children product_factors;
        const auto coefficient = subset.coefficients[i];
        if (coefficient != 1.0) {
            product_factors.push_back(std::make_shared<Scalar>(coefficient));
        }
        for (const auto& tensor : vertices.at(vertex).tensors) {
            product_factors.push_back(tensor);
        }
        summands.push_back(product_factors.size() == 1
                               ? product_factors.front()
                               : std::make_shared<Mul>(std::move(product_factors)));
    }
    return summands.size() == 1 ? summands.front() : std::make_shared<Add>(std::move(summands));
}

bool contains_term(const std::vector<NodePtr>& values, const NodePtr& needle) {
    return std::any_of(values.begin(), values.end(), [&](const NodePtr& value) {
        return value->structurally_equal(*needle);
    });
}

NodePtr replace_covered(const NodePtr& rhs, const std::vector<NodePtr>& covered,
                        const NodePtr& replacement, bool& changed) {
    auto expanded = terms(rhs);
    for (const auto& term : covered) {
        if (!contains_term(expanded, term)) {
            return rhs;
        }
    }
    Children remaining;
    for (const auto& term : expanded) {
        if (contains_term(covered, term)) {
            changed = true;
        } else {
            remaining.push_back(term);
        }
    }
    if (!changed) {
        return rhs;
    }
    remaining.push_back(replacement);
    return collect(std::make_shared<Add>(std::move(remaining)));
}

void validate_expression_axes(const Tensor& lhs, const NodePtr& rhs) {
    const auto rhs_indices = rhs->external_indices();
    const auto& lhs_indices = lhs.indices();
    if (rhs_indices.size() != lhs_indices.size()) {
        throw std::invalid_argument("rewritten expression changed its external index rank");
    }
    auto expected = lhs_indices;
    auto actual = rhs_indices;
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    if (expected != actual) {
        throw std::invalid_argument("rewritten expression changed its external indices");
    }
}

std::vector<Expression> canonicalized_inputs(
    const std::vector<Expression>& expressions,
    const CanonicalizationMetadata& canonicalization, bool& changed) {
    std::vector<Expression> result;
    result.reserve(expressions.size());
    changed = false;
    for (const auto& expression : expressions) {
        auto canonical = canonicalize_with_mapping(expression.rhs(), canonicalization);
        auto rhs = collect(canonical.node);
        auto lhs = std::dynamic_pointer_cast<Tensor>(
            map_indices(expression.lhs(), canonical.external_index_mapping));
        if (!lhs) {
            throw std::logic_error("canonicalizing an expression changed its lhs type");
        }
        changed = changed || !lhs->structurally_equal(*expression.lhs()) ||
                  !rhs->structurally_equal(*expression.rhs());
        validate_expression_axes(*lhs, rhs);
        result.emplace_back(std::move(lhs), std::move(rhs));
    }
    return result;
}

double total_cost(const std::vector<Expression>& expressions,
                  const IndexSpaces& spaces, const SpaceSizes& sizes,
                  const CostFactors& factors, const CostModel& model) {
    double total = 0.0;
    for (const auto& expression : expressions) {
        total += estimate_cost(expression.rhs(), spaces, sizes, model, factors).total;
    }
    return total;
}

NodePtr rename_symbols(const NodePtr& node,
                       const std::unordered_map<std::string, std::string>& mapping) {
    const auto renamed = [&](const NodePtr& item) -> NodePtr {
        const auto* tensor = dynamic_cast<const Tensor*>(item.get());
        if (!tensor) {
            return item;
        }
        const auto found = mapping.find(tensor->symbol());
        if (found == mapping.end() || found->second == tensor->symbol()) {
            return item;
        }
        return std::make_shared<Tensor>(found->second, tensor->indices());
    };
    return apply(node, renamed);
}

void deduplicate_intermediates(
    std::vector<Expression>& expressions,
    const std::unordered_set<std::string>& intermediate_symbols,
    const CanonicalizationMetadata& canonicalization) {
    struct Entry {
        std::vector<std::string> lhs_indices;
        NodePtr rhs;
        std::string symbol;
    };
    std::unordered_map<std::size_t, std::vector<Entry>> memo;
    std::unordered_map<std::string, std::string> aliases;
    std::vector<Expression> unique;
    unique.reserve(expressions.size());

    for (const auto& expression : expressions) {
        const auto rhs = rename_symbols(expression.rhs(), aliases);
        if (intermediate_symbols.find(expression.lhs()->symbol()) ==
            intermediate_symbols.end()) {
            unique.emplace_back(expression.lhs(), rhs);
            continue;
        }

        auto canonical = canonicalize_with_mapping(rhs, canonicalization);
        auto lhs_indices = expression.lhs()->indices();
        for (auto& index : lhs_indices) {
            const auto found = canonical.external_index_mapping.find(index);
            if (found != canonical.external_index_mapping.end()) {
                index = found->second;
            }
        }
        std::size_t hash = static_cast<std::size_t>(canonical.node->structural_hash());
        for (const auto& index : lhs_indices) {
            hash ^= std::hash<std::string>{}(index) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        auto& bucket = memo[hash];
        const auto found = std::find_if(bucket.begin(), bucket.end(), [&](const Entry& entry) {
            return entry.lhs_indices == lhs_indices &&
                   entry.rhs->structurally_equal(*canonical.node);
        });
        if (found != bucket.end()) {
            aliases.emplace(expression.lhs()->symbol(), found->symbol);
            continue;
        }
        bucket.push_back({std::move(lhs_indices), std::move(canonical.node),
                          expression.lhs()->symbol()});
        unique.emplace_back(expression.lhs(), rhs);
    }
    expressions = std::move(unique);
}

void cleanup_intermediates(std::vector<Expression>& intermediates,
                           std::vector<Expression>& expressions,
                           const std::string& format) {
    std::unordered_map<std::string, std::string> aliases;
    for (auto it = intermediates.rbegin(); it != intermediates.rend(); ++it) {
        const auto tensors = tensors_in(it->rhs());
        if (tensors.size() != 1 || std::abs(scalar_coefficient(it->rhs()) - 1.0) > 1e-12 ||
            tensors.front()->indices() != it->lhs()->indices()) {
            continue;
        }
        aliases[it->lhs()->symbol()] = tensors.front()->symbol();
    }
    if (!aliases.empty()) {
        for (auto& [old_symbol, target] : aliases) {
            std::unordered_set<std::string> visited{old_symbol};
            for (;;) {
                const auto found = aliases.find(target);
                if (found == aliases.end()) {
                    break;
                }
                if (!visited.insert(target).second) {
                    throw std::logic_error("cyclic aliases among CSE intermediates");
                }
                target = found->second;
            }
        }
        for (auto& expression : intermediates) {
            expression = Expression(expression.lhs(), rename_symbols(expression.rhs(), aliases));
        }
        for (auto& expression : expressions) {
            expression = Expression(expression.lhs(), rename_symbols(expression.rhs(), aliases));
        }
        intermediates.erase(
            std::remove_if(intermediates.begin(), intermediates.end(), [&](const auto& expression) {
                return aliases.find(expression.lhs()->symbol()) != aliases.end();
            }), intermediates.end());
    }

    std::vector<std::string> names;
    std::unordered_set<std::string> user_symbols;
    for (const auto& intermediate : intermediates) {
        names.push_back(intermediate.lhs()->symbol());
    }
    std::unordered_set<std::string> generated_symbols(names.begin(), names.end());
    for (const auto& expression : expressions) {
        user_symbols.insert(expression.lhs()->symbol());
        for (const auto& tensor : tensors_in(expression.rhs())) {
            if (generated_symbols.find(tensor->symbol()) == generated_symbols.end()) {
                user_symbols.insert(tensor->symbol());
            }
        }
    }
    std::sort(names.begin(), names.end());
    std::unordered_map<std::string, std::string> renumbered;
    std::size_t number = 0;
    for (const auto& name : names) {
        auto replacement = next_name(format, number++);
        while (user_symbols.find(replacement) != user_symbols.end()) {
            replacement = next_name(format, number++);
        }
        renumbered.emplace(name, std::move(replacement));
    }
    if (!renumbered.empty()) {
        auto rename_expression = [&](Expression& expression) {
            const auto lhs_symbol = renumbered.find(expression.lhs()->symbol());
            auto lhs = expression.lhs();
            if (lhs_symbol != renumbered.end()) {
                lhs = std::make_shared<Tensor>(lhs_symbol->second, lhs->indices());
            }
            expression = Expression(std::move(lhs), rename_symbols(expression.rhs(), renumbered));
        };
        for (auto& expression : intermediates) {
            rename_expression(expression);
        }
        for (auto& expression : expressions) {
            rename_expression(expression);
        }
    }
}

}  // namespace

namespace {

std::vector<Expression> common_subexpression_elimination_impl(
    std::vector<Expression> expressions, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostFactors& cost_factors,
    const CanonicalizationMetadata& canonicalization, const CSEOptions& options,
    bool parenthesize_result) {
    CSEDepthGuard depth_guard;
    if (expressions.empty()) {
        return {};
    }
    for (auto& expression : expressions) {
        validate_expression_axes(*expression.lhs(), expression.rhs());
    }
    ParenthesizationCache path_cache;
    std::vector<Expression> intermediates;
    std::unordered_set<std::string> reserved_symbols;
    std::size_t nested_format_serial = 0;
    for (const auto& expression : expressions) {
        reserved_symbols.insert(expression.lhs()->symbol());
        for (const auto& tensor : tensors_in(expression.rhs())) {
            reserved_symbols.insert(tensor->symbol());
        }
    }

    ParenthesizationCache trial_path_cache;
    for (;;) {
        std::vector<Expression> all = intermediates;
        all.insert(all.end(), expressions.begin(), expressions.end());
        const auto graphs = build_graphs(all, spaces, sizes, options, path_cache);
        auto candidates = select_bicliques(graphs, spaces, sizes, cost_factors,
                                           options.cost_model, options.max_bicliques);
        if (candidates.empty()) {
            break;
        }
        bool accepted = false;
        double best_cost = std::numeric_limits<double>::infinity();
        std::vector<Expression> best_expressions;
        std::vector<Expression> best_intermediates;
        for (const auto& candidate : candidates) {
            const auto& graph = graphs.at(candidate.graph_index).graph;
            const auto covered = covered_terms(graph, candidate.biclique);
            if (covered.empty()) {
                continue;
            }
            auto left_rhs = build_side(graph.left_vertices(), candidate.biclique.left);
            auto right_rhs = build_side(graph.right_vertices(), candidate.biclique.right);
            auto left_indices = left_rhs->external_indices();
            auto right_indices = right_rhs->external_indices();
            auto left_name = next_available_name(options.intermediate_format,
                                                 reserved_symbols);
            reserved_symbols.insert(left_name);
            auto right_name = next_available_name(options.intermediate_format,
                                                  reserved_symbols);
            reserved_symbols.insert(right_name);
            auto left_tensor = std::make_shared<Tensor>(left_name, left_indices);
            auto right_tensor = std::make_shared<Tensor>(right_name, right_indices);
            Expression left_expression(left_tensor, left_rhs);
            Expression right_expression(right_tensor, right_rhs);
            const auto reuse_definition = [&](const Expression& candidate_expression)
                -> std::shared_ptr<Tensor> {
                for (const auto& existing : all) {
                    if (existing.lhs()->indices() == candidate_expression.lhs()->indices() &&
                        existing.rhs()->structurally_equal(*candidate_expression.rhs())) {
                        return existing.lhs();
                    }
                }
                return {};
            };
            auto reused_left = reuse_definition(left_expression);
            if (reused_left) {
                left_tensor = reused_left;
            }
            auto reused_right = reuse_definition(right_expression);
            if (reused_right) {
                right_tensor = reused_right;
            }
            const bool add_left = !reused_left;
            const bool add_right = !reused_right;

            std::vector<Expression> nested_intermediates;
            if (add_left || add_right) {
                std::vector<Expression> recursive_inputs;
                if (add_left) {
                    recursive_inputs.push_back(left_expression);
                }
                if (add_right) {
                    recursive_inputs.push_back(right_expression);
                }
                auto recursive_options = options;
                do {
                    recursive_options.intermediate_format =
                        "__cse_nested" + std::to_string(cse_depth) + "_" +
                        std::to_string(nested_format_serial++) + "_{}__";
                } while (reserved_symbols.find(
                             next_name(recursive_options.intermediate_format, 0)) !=
                         reserved_symbols.end());
                std::vector<Expression> recursively_optimized;
                try {
                    recursively_optimized = common_subexpression_elimination_impl(
                        std::move(recursive_inputs), spaces, sizes, cost_factors,
                        canonicalization, recursive_options, false);
                } catch (const std::exception& error) {
                    std::throw_with_nested(std::runtime_error(
                        std::string("recursive CSE failed: ") + error.what()));
                }
                for (const auto& expression : recursively_optimized) {
                    if (expression.lhs()->symbol() == left_name) {
                        std::unordered_map<std::string, std::string> restore_indices;
                        for (std::size_t i = 0; i < left_indices.size(); ++i) {
                            restore_indices.emplace(expression.lhs()->indices()[i],
                                                    left_indices[i]);
                        }
                        left_tensor = std::make_shared<Tensor>(left_name, left_indices);
                        left_expression = Expression(
                            left_tensor, map_indices(expression.rhs(), restore_indices));
                    } else if (expression.lhs()->symbol() == right_name) {
                        std::unordered_map<std::string, std::string> restore_indices;
                        for (std::size_t i = 0; i < right_indices.size(); ++i) {
                            restore_indices.emplace(expression.lhs()->indices()[i],
                                                    right_indices[i]);
                        }
                        right_tensor = std::make_shared<Tensor>(right_name, right_indices);
                        right_expression = Expression(
                            right_tensor, map_indices(expression.rhs(), restore_indices));
                    } else {
                        nested_intermediates.push_back(expression);
                    }
                }
            }
            auto replacement = std::make_shared<Mul>(Children{
                left_tensor, right_tensor,
                std::make_shared<Scalar>(candidate.biclique.coefficient)});

            auto trial_all = all;
            bool changed_any = false;
            for (auto& expression : trial_all) {
                bool changed = false;
                auto rhs = replace_covered(expression.rhs(), covered, replacement, changed);
                if (changed) {
                    validate_expression_axes(*expression.lhs(), rhs);
                    expression = Expression(expression.lhs(), std::move(rhs));
                    changed_any = true;
                }
            }
            if (!changed_any) {
                continue;
            }
            trial_all.insert(trial_all.end(), nested_intermediates.begin(),
                             nested_intermediates.end());
            std::unordered_set<std::string> trial_intermediate_symbols;
            for (const auto& expression : intermediates) {
                trial_intermediate_symbols.insert(expression.lhs()->symbol());
            }
            for (const auto& expression : nested_intermediates) {
                trial_intermediate_symbols.insert(expression.lhs()->symbol());
            }
            if (add_left) {
                trial_intermediate_symbols.insert(left_expression.lhs()->symbol());
                trial_all.push_back(std::move(left_expression));
            }
            if (add_right) {
                trial_intermediate_symbols.insert(right_expression.lhs()->symbol());
                trial_all.push_back(std::move(right_expression));
            }
            trial_all = topological_sort(std::move(trial_all));
            deduplicate_intermediates(trial_all, trial_intermediate_symbols,
                                      canonicalization);
            trial_all = topological_sort(std::move(trial_all));
            auto cost_trial = trial_all;
            for (auto& expression : cost_trial) {
                auto rhs = parenthesize_products(expression.rhs(), spaces, sizes,
                                                 cost_factors, options,
                                                 trial_path_cache);
                expression = Expression(expression.lhs(), std::move(rhs));
            }
            const double trial_cost = total_cost(cost_trial, spaces, sizes, cost_factors,
                                                 options.cost_model);
            if (trial_cost < best_cost) {
                best_cost = trial_cost;
                best_intermediates.clear();
                best_expressions.clear();
                for (const auto& expression : trial_all) {
                    if (trial_intermediate_symbols.find(expression.lhs()->symbol()) !=
                        trial_intermediate_symbols.end()) {
                        best_intermediates.push_back(expression);
                    } else {
                        best_expressions.push_back(expression);
                    }
                }
                accepted = true;
            }
        }
        if (!accepted) {
            break;
        }
        expressions = std::move(best_expressions);
        intermediates = std::move(best_intermediates);
        for (const auto& expression : intermediates) {
            reserved_symbols.insert(expression.lhs()->symbol());
        }
    }

    if (parenthesize_result) {
        ParenthesizationCache path_cache;
        for (auto& expression : intermediates) {
            auto rhs = parenthesize_products(expression.rhs(), spaces, sizes,
                                             cost_factors, options, path_cache);
            expression = Expression(expression.lhs(), std::move(rhs));
        }
        for (auto& expression : expressions) {
            auto rhs = parenthesize_products(expression.rhs(), spaces, sizes,
                                             cost_factors, options, path_cache);
            expression = Expression(expression.lhs(), std::move(rhs));
        }
    }
    cleanup_intermediates(intermediates, expressions, options.intermediate_format);
    std::unordered_set<std::string> intermediate_symbols;
    for (const auto& expression : intermediates) {
        intermediate_symbols.insert(expression.lhs()->symbol());
    }
    std::vector<Expression> result = std::move(intermediates);
    result.insert(result.end(), expressions.begin(), expressions.end());
    result = topological_sort(std::move(result));
    deduplicate_intermediates(result, intermediate_symbols, canonicalization);
    result = topological_sort(std::move(result));
    return result;
}

}  // namespace

std::vector<Expression> common_subexpression_elimination(
    std::vector<Expression> expressions, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostFactors& cost_factors,
    const CanonicalizationMetadata& canonicalization, const CSEOptions& options) {
    auto resolved_options = options;
    if (!resolved_options.custom_path_requests) {
        const bool constrained = !options.scaling_limits.time.empty() ||
                                 !options.scaling_limits.memory.empty();
        PathRequest small_request;
        small_request.exhaustive = constrained
            ? options.max_path_samples
            : options.max_path_samples > 0 ? options.max_path_samples - 1 : 0;
        small_request.dynamic = !constrained && options.max_path_samples > 0;
        if (options.exhaustive_limit == 0) {
            resolved_options.path_requests = {{}, {{0, 0, 0, true}}};
        } else {
            resolved_options.path_requests = {
                {options.exhaustive_limit},
                {small_request, PathRequest{0, 0, 0, true}}};
        }
    }
    auto result = common_subexpression_elimination_impl(
        expressions, spaces, sizes, cost_factors,
        canonicalization, resolved_options, true);
    if (!resolved_options.scaling_limits.time.empty() ||
        !resolved_options.scaling_limits.memory.empty()) {
        return result;
    }
    bool changed = false;
    auto normalized = canonicalized_inputs(expressions, canonicalization, changed);
    if (!changed) {
        return result;
    }
    auto normalized_result = common_subexpression_elimination_impl(
        std::move(normalized), spaces, sizes, cost_factors,
        canonicalization, resolved_options, true);
    const auto result_cost = total_cost(result, spaces, sizes, cost_factors,
                                        resolved_options.cost_model);
    const auto normalized_cost = total_cost(normalized_result, spaces, sizes, cost_factors,
                                            resolved_options.cost_model);
    return normalized_cost < result_cost ? normalized_result : result;
}

}  // namespace albert
