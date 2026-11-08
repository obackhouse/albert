#include "parenthesize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace albert {
namespace {

using Axes = std::set<std::string>;

Axes get_axes(const NodePtr& node) {
    const auto indices = node->external_indices();
    return Axes(indices.begin(), indices.end());
}

Axes unite(const Axes& lhs, const Axes& rhs) {
    Axes result = lhs;
    result.insert(rhs.begin(), rhs.end());
    return result;
}

Axes xor_axes(const Axes& lhs, const Axes& rhs) {
    Axes result;
    std::set_symmetric_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                                  std::inserter(result, result.end()));
    return result;
}

bool shares_axis(const Axes& lhs, const Axes& rhs) {
    auto left = lhs.begin();
    auto right = rhs.begin();
    while (left != lhs.end() && right != rhs.end()) {
        if (*left == *right) {
            return true;
        }
        if (*left < *right) {
            ++left;
        } else {
            ++right;
        }
    }
    return false;
}

class LimitsChecker {
public:
    LimitsChecker(const IndexSpaces& spaces, const SpaceSizes& sizes,
                  const ScalingLimits& limits)
        : sizes_(sizes), limits_(limits) {
        for (const auto& [space, indices] : spaces) {
            if (sizes_.find(space) == sizes_.end()) {
                throw std::invalid_argument("space " + space + " has no declared size");
            }
            for (const auto& index : indices) {
                const auto [it, inserted] = index_spaces_.emplace(index, space);
                if (!inserted && it->second != space) {
                    throw std::invalid_argument("an index cannot belong to multiple spaces");
                }
            }
        }
        for (const auto& [space, size] : sizes_) {
            if (size == 0) {
                throw std::invalid_argument("space sizes must be positive");
            }
            static_cast<void>(space);
        }
    }

    bool allows(const Axes& operation, const Axes& output) const {
        return under_caps(operation, limits_.time) && under_caps(output, limits_.memory);
    }

    bool memory_allows_tree(const NodePtr& node) const {
        if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
            const Axes axes(tensor->indices().begin(), tensor->indices().end());
            return under_caps(axes, limits_.memory);
        }
        for (const auto& child : node->children()) {
            if (!memory_allows_tree(child)) {
                return false;
            }
        }
        if (node->kind() == NodeKind::Scalar) {
            return true;
        }
        const auto indices = node->external_indices();
        return under_caps(Axes(indices.begin(), indices.end()), limits_.memory);
    }

    bool time_allows_tree(const NodePtr& node) const {
        if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
            const Axes all_axes(tensor->indices().begin(), tensor->indices().end());
            const auto external_indices = tensor->external_indices();
            const Axes external(external_indices.begin(), external_indices.end());
            return all_axes.size() == external.size() || under_caps(all_axes, limits_.time);
        }
        for (const auto& child : node->children()) {
            if (!time_allows_tree(child)) {
                return false;
            }
        }
        if (node->kind() != NodeKind::Mul || node->children().size() < 2) {
            return true;
        }
        auto axes = get_axes(node->children().front());
        for (std::size_t i = 1; i < node->children().size(); ++i) {
            const auto right = get_axes(node->children()[i]);
            const auto operation = unite(axes, right);
            if (!under_caps(operation, limits_.time)) {
                return false;
            }
            axes = xor_axes(axes, right);
        }
        return true;
    }

    double size(const Axes& axes) const {
        double result = 1.0;
        for (const auto& index : axes) {
            const auto space = index_spaces_.find(index);
            if (space == index_spaces_.end()) {
                throw std::invalid_argument("index " + index + " does not belong to a space");
            }
            result *= static_cast<double>(sizes_.at(space->second));
            if (!std::isfinite(result)) {
                throw std::overflow_error("estimated tensor size exceeds numeric range");
            }
        }
        return result;
    }

private:
    bool under_caps(const Axes& axes, const ScalingCaps& caps) const {
        for (const auto& [cap_spaces, maximum] : caps) {
            std::size_t count = 0;
            for (const auto& index : axes) {
                const auto space = index_spaces_.find(index);
                if (space == index_spaces_.end()) {
                    throw std::invalid_argument("index " + index + " does not belong to a space");
                }
                if (std::find(cap_spaces.begin(), cap_spaces.end(), space->second) !=
                    cap_spaces.end()) {
                    ++count;
                }
            }
            if (count > maximum) {
                return false;
            }
        }
        return true;
    }

    const SpaceSizes& sizes_;
    const ScalingLimits& limits_;
    std::unordered_map<std::string, std::string> index_spaces_;
};

struct Active {
    NodePtr node;
    Axes axes;
};

ParenthesizationCandidate make_candidate(const std::vector<Contraction>& path,
                                          const Mul& source,
                                          const IndexSpaces& spaces,
                                          const SpaceSizes& sizes,
                                          const CostModel& model) {
    ParenthesizationCandidate candidate;
    candidate.path = path;
    const auto applied = apply_parenthesization(source, candidate);
    candidate.cost = estimate_cost(applied, spaces, sizes, model);
    return candidate;
}

std::vector<ParenthesizationCandidate> exhaustive(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model, std::size_t maximum) {
    std::vector<ParenthesizationCandidate> best;
    std::vector<Contraction> path;
    std::function<void(const std::vector<Active>&)> visit;
    visit = [&](const std::vector<Active>& active) {
        if (active.size() == 1) {
            auto candidate = make_candidate(path, node, spaces, sizes, model);
            best.push_back(std::move(candidate));
            std::stable_sort(best.begin(), best.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.cost.total < rhs.cost.total;
            });
            if (best.size() > maximum) {
                best.pop_back();
            }
            return;
        }
        for (std::size_t i = 0; i < active.size(); ++i) {
            for (std::size_t j = i + 1; j < active.size(); ++j) {
                const auto operation = unite(active[i].axes, active[j].axes);
                const auto output = xor_axes(active[i].axes, active[j].axes);
                if (!checker.allows(operation, output)) {
                    continue;
                }
                std::vector<Active> next = active;
                next[i] = {std::make_shared<Mul>(Children{active[i].node, active[j].node}), output};
                next.erase(next.begin() + static_cast<std::ptrdiff_t>(j));
                path.emplace_back(i, j);
                visit(next);
                path.pop_back();
            }
        }
    };

    std::vector<Active> active;
    active.reserve(node.children().size());
    for (const auto& child : node.children()) {
        active.push_back({child, get_axes(child)});
    }
    visit(active);
    return best;
}

struct DpEntry {
    double score = std::numeric_limits<double>::infinity();
    std::uint64_t left = 0;
    std::uint64_t right = 0;
    Axes axes;
};

std::vector<ParenthesizationCandidate> dynamic_programming(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model) {
    const auto count = node.children().size();
    if (count > 63) {
        throw std::invalid_argument("DP parenthesization supports at most 63 operands");
    }
    const std::uint64_t full = (std::uint64_t{1} << count) - 1;
    std::vector<Axes> leaf_axes;
    leaf_axes.reserve(count);
    for (const auto& child : node.children()) {
        leaf_axes.push_back(get_axes(child));
    }
    std::unordered_map<std::uint64_t, DpEntry> memo;
    std::function<const DpEntry&(std::uint64_t)> solve;
    solve = [&](std::uint64_t mask) -> const DpEntry& {
        const auto found = memo.find(mask);
        if (found != memo.end()) {
            return found->second;
        }
        DpEntry entry;
        if ((mask & (mask - 1)) == 0) {
            const auto index = static_cast<std::size_t>(__builtin_ctzll(mask));
            entry.score = 0.0;
            entry.axes = leaf_axes[index];
            return memo.emplace(mask, std::move(entry)).first->second;
        }
        const auto first_bit = mask & (~mask + 1);
        for (std::uint64_t left = (mask - 1) & mask; left; left = (left - 1) & mask) {
            if (!(left & first_bit)) {
                continue;
            }
            const auto right = mask ^ left;
            if (right == 0) {
                continue;
            }
            const auto& lhs = solve(left);
            const auto& rhs = solve(right);
            if (!shares_axis(lhs.axes, rhs.axes)) {
                continue;
            }
            const auto operation = unite(lhs.axes, rhs.axes);
            const auto output = xor_axes(lhs.axes, rhs.axes);
            if (!checker.allows(operation, output)) {
                continue;
            }
            const auto score = lhs.score + rhs.score + checker.size(operation);
            if (score < entry.score) {
                entry.score = score;
                entry.left = left;
                entry.right = right;
                entry.axes = output;
            }
        }
        return memo.emplace(mask, std::move(entry)).first->second;
    };
    const auto& root = solve(full);
    if (!std::isfinite(root.score)) {
        return {};
    }

    std::vector<Contraction> path;
    std::vector<std::uint64_t> active;
    for (std::size_t i = 0; i < count; ++i) {
        active.push_back(std::uint64_t{1} << i);
    }
    std::function<void(std::uint64_t)> emit = [&](std::uint64_t mask) {
        const auto found = memo.find(mask);
        if (found == memo.end() || found->second.left == 0) {
            return;
        }
        const auto left = found->second.left;
        const auto right = found->second.right;
        emit(left);
        emit(right);
        const auto left_pos = std::find(active.begin(), active.end(), left);
        const auto right_pos = std::find(active.begin(), active.end(), right);
        if (left_pos == active.end() || right_pos == active.end()) {
            throw std::logic_error("invalid DP parenthesization state");
        }
        auto i = static_cast<std::size_t>(left_pos - active.begin());
        auto j = static_cast<std::size_t>(right_pos - active.begin());
        if (i > j) {
            std::swap(i, j);
        }
        path.emplace_back(i, j);
        active[i] = left | right;
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
    };
    emit(full);
    return {make_candidate(path, node, spaces, sizes, model)};
}

}  // namespace

std::vector<ParenthesizationCandidate> parenthesization_candidates(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits, std::size_t max_candidates,
    std::size_t exhaustive_limit, const CostModel& model) {
    if (node.children().empty()) {
        throw std::invalid_argument("cannot parenthesize an empty Mul");
    }
    LimitsChecker checker(spaces, sizes, limits);
    for (const auto& child : node.children()) {
        if (!checker.memory_allows_tree(child) || !checker.time_allows_tree(child)) {
            return {};
        }
    }
    if (node.children().size() > exhaustive_limit) {
        return dynamic_programming(node, checker, spaces, sizes, model);
    }
    if (max_candidates == 0) {
        return {};
    }
    return exhaustive(node, checker, spaces, sizes, model, max_candidates);
}

NodePtr apply_parenthesization(const Mul& node,
                               const ParenthesizationCandidate& candidate) {
    std::vector<NodePtr> active(node.children().begin(), node.children().end());
    for (const auto& [i, j] : candidate.path) {
        if (i >= active.size() || j >= active.size() || i == j) {
            throw std::invalid_argument("candidate path contains invalid contraction positions");
        }
        const auto first = std::min(i, j);
        const auto second = std::max(i, j);
        auto result = std::make_shared<Mul>(Children{active[first], active[second]});
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(second));
        active[first] = std::move(result);
    }
    if (active.size() != 1) {
        throw std::invalid_argument("candidate path must contract all Mul operands");
    }
    return active.front();
}

}  // namespace albert
