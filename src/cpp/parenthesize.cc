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
                                          const CostModel& model,
                                          const CostFactors& cost_factors) {
    ParenthesizationCandidate candidate;
    candidate.path = path;
    const auto applied = apply_parenthesization(source, candidate);
    candidate.cost = estimate_cost(applied, spaces, sizes, model, cost_factors);
    return candidate;
}

std::vector<ParenthesizationCandidate> exhaustive(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model,
    const CostFactors& cost_factors, std::size_t maximum) {
    std::vector<ParenthesizationCandidate> best;
    std::vector<Contraction> path;
    std::function<void(const std::vector<Active>&)> visit;
    visit = [&](const std::vector<Active>& active) {
        if (active.size() == 1) {
            auto candidate = make_candidate(path, node, spaces, sizes, model, cost_factors);
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
                next.erase(next.begin() + static_cast<std::ptrdiff_t>(j));
                next.erase(next.begin() + static_cast<std::ptrdiff_t>(i));
                next.push_back({std::make_shared<Mul>(Children{active[i].node, active[j].node}),
                                output});
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
};

struct BitmaskDpEntry {
    double score = std::numeric_limits<double>::infinity();
    std::uint64_t left = 0;
    std::uint64_t right = 0;
    bool computed = false;
};

struct MaskCap {
    std::uint64_t axes = 0;
    std::size_t maximum = 0;
};

// Dense axis masks avoid set construction for operands with at most 64 distinct indices.
bool dynamic_programming_bitmask(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits, const CostModel& model,
    const CostFactors& cost_factors,
    std::vector<ParenthesizationCandidate>& result) {
    const auto count = node.children().size();
    if (count > 20) {
        return false;
    }
    std::set<std::string> unique_indices;
    for (const auto& child : node.children()) {
        const auto external = child->external_indices();
        unique_indices.insert(external.begin(), external.end());
    }
    if (unique_indices.size() > 64) {
        return false;
    }

    std::vector<std::string> axis_names(unique_indices.begin(), unique_indices.end());
    std::unordered_map<std::string, std::size_t> axis_positions;
    std::vector<double> axis_sizes;
    axis_sizes.reserve(axis_names.size());
    for (std::size_t axis = 0; axis < axis_names.size(); ++axis) {
        axis_positions.emplace(axis_names[axis], axis);
        bool found_space = false;
        for (const auto& [space, indices] : spaces) {
            if (std::find(indices.begin(), indices.end(), axis_names[axis]) == indices.end()) {
                continue;
            }
            const auto size = sizes.find(space);
            if (size == sizes.end()) {
                throw std::invalid_argument("space " + space + " has no declared size");
            }
            axis_sizes.push_back(static_cast<double>(size->second));
            found_space = true;
            break;
        }
        if (!found_space) {
            throw std::invalid_argument("index " + axis_names[axis] + " does not belong to a space");
        }
    }
    const auto make_caps = [&](const ScalingCaps& caps) {
        std::vector<MaskCap> result_caps;
        result_caps.reserve(caps.size());
        for (const auto& [cap_spaces, maximum] : caps) {
            std::uint64_t cap_axes = 0;
            for (std::size_t axis = 0; axis < axis_names.size(); ++axis) {
                for (const auto& [space, indices] : spaces) {
                    if (std::find(indices.begin(), indices.end(), axis_names[axis]) !=
                            indices.end() &&
                        std::find(cap_spaces.begin(), cap_spaces.end(), space) !=
                            cap_spaces.end()) {
                        cap_axes |= std::uint64_t{1} << axis;
                        break;
                    }
                }
            }
            result_caps.push_back({cap_axes, maximum});
        }
        return result_caps;
    };
    const auto time_caps = make_caps(limits.time);
    const auto memory_caps = make_caps(limits.memory);
    const auto under_caps = [](std::uint64_t axes, const std::vector<MaskCap>& caps) {
        for (const auto& cap : caps) {
            if (static_cast<std::size_t>(__builtin_popcountll(axes & cap.axes)) >
                cap.maximum) {
                return false;
            }
        }
        return true;
    };
    const auto axes_size = [&](std::uint64_t axes) {
        double product = 1.0;
        while (axes != 0) {
            const auto bit = static_cast<std::size_t>(__builtin_ctzll(axes));
            product *= axis_sizes[bit];
            axes &= axes - 1;
        }
        return product;
    };

    std::vector<std::uint64_t> leaf_axes;
    leaf_axes.reserve(count);
    for (const auto& child : node.children()) {
        std::uint64_t axes = 0;
        for (const auto& index : child->external_indices()) {
            axes |= std::uint64_t{1} << axis_positions.at(index);
        }
        leaf_axes.push_back(axes);
    }
    const auto state_count = std::size_t{1} << count;
    std::vector<BitmaskDpEntry> memo(state_count);
    std::vector<std::uint64_t> subset_axes_memo(state_count);
    std::vector<bool> subset_axes_computed(state_count, false);
    std::function<std::uint64_t(std::uint64_t)> subset_axes;
    subset_axes = [&](std::uint64_t mask) {
        if (subset_axes_computed[mask]) {
            return subset_axes_memo[mask];
        }
        const auto first_bit = mask & (~mask + 1);
        std::uint64_t axes;
        if (mask == first_bit) {
            axes = leaf_axes[static_cast<std::size_t>(__builtin_ctzll(mask))];
        } else {
            axes = leaf_axes[static_cast<std::size_t>(__builtin_ctzll(first_bit))] ^
                   subset_axes(mask ^ first_bit);
        }
        subset_axes_computed[mask] = true;
        subset_axes_memo[mask] = axes;
        return axes;
    };
    std::function<const BitmaskDpEntry&(std::uint64_t)> solve;
    solve = [&](std::uint64_t mask) -> const BitmaskDpEntry& {
        auto& entry = memo[mask];
        if (entry.computed) {
            return entry;
        }
        entry.computed = true;
        if ((mask & (mask - 1)) == 0) {
            entry.score = 0.0;
            return entry;
        }
        std::vector<std::size_t> positions;
        for (std::size_t i = 0; i < count; ++i) {
            if (mask & (std::uint64_t{1} << i)) {
                positions.push_back(i);
            }
        }
        const auto subset_count = positions.size();
        for (std::size_t left_count = 1; left_count <= subset_count / 2; ++left_count) {
            std::vector<std::size_t> combination(left_count);
            for (std::size_t i = 0; i < left_count; ++i) {
                combination[i] = i;
            }
            for (;;) {
                std::uint64_t left = 0;
                for (const auto position : combination) {
                    left |= std::uint64_t{1} << positions[position];
                }
                const auto right = mask ^ left;
                if ((left_count * 2 != subset_count || left < right) && right != 0) {
                    const auto left_axes = subset_axes(left);
                    const auto right_axes = subset_axes(right);
                    if ((left_axes & right_axes) != 0) {
                        const auto operation = left_axes | right_axes;
                        const auto output = left_axes ^ right_axes;
                        if (under_caps(operation, time_caps) &&
                            under_caps(output, memory_caps)) {
                            const auto& lhs = solve(left);
                            const auto& rhs = solve(right);
                            if (std::isfinite(lhs.score) && std::isfinite(rhs.score)) {
                                const auto score = lhs.score + rhs.score + axes_size(operation);
                                if (score < entry.score) {
                                    entry.score = score;
                                    entry.left = left;
                                    entry.right = right;
                                }
                            }
                        }
                    }
                }
                std::size_t position = left_count;
                bool advanced = false;
                while (position > 0) {
                    --position;
                    if (combination[position] < subset_count - left_count + position) {
                        ++combination[position];
                        for (std::size_t j = position + 1; j < left_count; ++j) {
                            combination[j] = combination[j - 1] + 1;
                        }
                        advanced = true;
                        break;
                    }
                }
                if (!advanced) {
                    break;
                }
            }
        }
        return entry;
    };

    std::vector<std::uint64_t> active;
    active.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        active.push_back(std::uint64_t{1} << i);
    }
    std::vector<Contraction> path;
    std::function<void(std::uint64_t)> emit = [&](std::uint64_t mask) {
        const auto& entry = solve(mask);
        if (entry.left == 0) {
            return;
        }
        emit(entry.left);
        emit(entry.right);
        const auto left_pos = std::find(active.begin(), active.end(), entry.left);
        const auto right_pos = std::find(active.begin(), active.end(), entry.right);
        if (left_pos == active.end() || right_pos == active.end()) {
            throw std::logic_error("invalid DP parenthesization state");
        }
        auto i = static_cast<std::size_t>(left_pos - active.begin());
        auto j = static_cast<std::size_t>(right_pos - active.begin());
        if (i > j) {
            std::swap(i, j);
        }
        path.emplace_back(i, j);
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
        active.push_back(mask);
    };

    std::vector<std::uint64_t> components;
    std::vector<std::uint64_t> unassigned;
    for (std::size_t i = count; i > 0; --i) {
        unassigned.push_back(std::uint64_t{1} << (i - 1));
    }
    while (!unassigned.empty()) {
        auto component = unassigned.back();
        unassigned.pop_back();
        bool grew = true;
        while (grew) {
            grew = false;
            for (auto it = unassigned.begin(); it != unassigned.end();) {
                if ((subset_axes(component) & subset_axes(*it)) != 0) {
                    component |= *it;
                    it = unassigned.erase(it);
                    grew = true;
                } else {
                    ++it;
                }
            }
        }
        components.push_back(component);
    }
    for (const auto component : components) {
        if (!std::isfinite(solve(component).score)) {
            return true;
        }
        emit(component);
    }
    std::stable_sort(components.begin(), components.end(), [&](auto lhs, auto rhs) {
        return axes_size(subset_axes(lhs)) < axes_size(subset_axes(rhs));
    });
    auto accumulated = components.front();
    for (std::size_t k = 1; k < components.size(); ++k) {
        const auto next = components[k];
        const auto operation = subset_axes(accumulated) | subset_axes(next);
        const auto output = subset_axes(accumulated) ^ subset_axes(next);
        if (!under_caps(operation, time_caps) || !under_caps(output, memory_caps)) {
            return true;
        }
        const auto left_pos = std::find(active.begin(), active.end(), accumulated);
        const auto right_pos = std::find(active.begin(), active.end(), next);
        if (left_pos == active.end() || right_pos == active.end()) {
            throw std::logic_error("invalid disconnected-component parenthesization state");
        }
        auto i = static_cast<std::size_t>(left_pos - active.begin());
        auto j = static_cast<std::size_t>(right_pos - active.begin());
        if (i > j) {
            std::swap(i, j);
        }
        path.emplace_back(i, j);
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
        accumulated |= next;
        active.push_back(accumulated);
    }
    result.push_back(make_candidate(path, node, spaces, sizes, model, cost_factors));
    return true;
}

std::vector<ParenthesizationCandidate> dynamic_programming_generic(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model,
    const CostFactors& cost_factors) {
    const auto count = node.children().size();
    if (count > 63) {
        throw std::invalid_argument("DP parenthesization supports at most 63 operands");
    }
    std::vector<Axes> leaf_axes;
    leaf_axes.reserve(count);
    for (const auto& child : node.children()) {
        leaf_axes.push_back(get_axes(child));
    }
    std::unordered_map<std::uint64_t, DpEntry> memo;
    std::unordered_map<std::uint64_t, Axes> subset_axes_memo;
    std::function<const Axes&(std::uint64_t)> subset_axes;
    subset_axes = [&](std::uint64_t mask) -> const Axes& {
        const auto found = subset_axes_memo.find(mask);
        if (found != subset_axes_memo.end()) {
            return found->second;
        }
        const auto first_bit = mask & (~mask + 1);
        Axes axes;
        if (mask == first_bit) {
            const auto index = static_cast<std::size_t>(__builtin_ctzll(mask));
            axes = leaf_axes[index];
        } else {
            axes = xor_axes(leaf_axes[static_cast<std::size_t>(__builtin_ctzll(first_bit))],
                            subset_axes(mask ^ first_bit));
        }
        return subset_axes_memo.emplace(mask, std::move(axes)).first->second;
    };
    std::function<const DpEntry&(std::uint64_t)> solve;
    solve = [&](std::uint64_t mask) -> const DpEntry& {
        const auto found = memo.find(mask);
        if (found != memo.end()) {
            return found->second;
        }
        DpEntry entry;
        if ((mask & (mask - 1)) == 0) {
            entry.score = 0.0;
            return memo.emplace(mask, std::move(entry)).first->second;
        }
        std::vector<std::size_t> positions;
        for (std::size_t i = 0; i < count; ++i) {
            if (mask & (std::uint64_t{1} << i)) {
                positions.push_back(i);
            }
        }
        const auto consider_partition = [&](std::uint64_t left) {
            const auto right = mask ^ left;
            if (right == 0 || (left == right)) {
                return;
            }
            const auto& left_axes = subset_axes(left);
            const auto& right_axes = subset_axes(right);
            if (!shares_axis(left_axes, right_axes)) {
                return;
            }
            const auto operation = unite(left_axes, right_axes);
            const auto output = xor_axes(left_axes, right_axes);
            if (!checker.allows(operation, output)) {
                return;
            }
            const auto lhs_score = solve(left).score;
            const auto rhs_score = solve(right).score;
            if (!std::isfinite(lhs_score) || !std::isfinite(rhs_score)) {
                return;
            }
            const auto score = lhs_score + rhs_score + checker.size(operation);
            if (score < entry.score) {
                entry.score = score;
                entry.left = left;
                entry.right = right;
            }
        };
        const auto subset_count = positions.size();
        for (std::size_t left_count = 1; left_count <= subset_count / 2; ++left_count) {
            std::vector<std::size_t> combination(left_count);
            for (std::size_t i = 0; i < left_count; ++i) {
                combination[i] = i;
            }
            for (;;) {
                std::uint64_t left = 0;
                for (const auto position : combination) {
                    left |= std::uint64_t{1} << positions[position];
                }
                if (left_count * 2 != subset_count || left < (mask ^ left)) {
                    consider_partition(left);
                }
                std::size_t position = left_count;
                bool advanced = false;
                while (position > 0) {
                    --position;
                    if (combination[position] < subset_count - left_count + position) {
                        ++combination[position];
                        for (std::size_t j = position + 1; j < left_count; ++j) {
                            combination[j] = combination[j - 1] + 1;
                        }
                        advanced = true;
                        break;
                    }
                }
                if (!advanced) {
                    break;
                }
            }
        }
        return memo.emplace(mask, std::move(entry)).first->second;
    };
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
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
        active.push_back(left | right);
    };
    std::vector<std::uint64_t> components;
    std::vector<std::uint64_t> unassigned;
    for (std::size_t i = count; i > 0; --i) {
        unassigned.push_back(std::uint64_t{1} << (i - 1));
    }
    while (!unassigned.empty()) {
        auto component = unassigned.back();
        unassigned.pop_back();
        bool grew = true;
        while (grew) {
            grew = false;
            for (auto it = unassigned.begin(); it != unassigned.end();) {
                if (shares_axis(subset_axes(component), subset_axes(*it))) {
                    component |= *it;
                    it = unassigned.erase(it);
                    grew = true;
                } else {
                    ++it;
                }
            }
        }
        components.push_back(component);
    }
    for (const auto component : components) {
        const auto& root = solve(component);
        if (!std::isfinite(root.score)) {
            return {};
        }
        emit(component);
    }
    std::stable_sort(components.begin(), components.end(), [&](auto lhs, auto rhs) {
        return checker.size(subset_axes(lhs)) < checker.size(subset_axes(rhs));
    });
    auto accumulated = components.front();
    for (std::size_t k = 1; k < components.size(); ++k) {
        const auto next = components[k];
        const auto operation = unite(subset_axes(accumulated), subset_axes(next));
        const auto output = xor_axes(subset_axes(accumulated), subset_axes(next));
        if (!checker.allows(operation, output)) {
            return {};
        }
        const auto left_pos = std::find(active.begin(), active.end(), accumulated);
        const auto right_pos = std::find(active.begin(), active.end(), next);
        if (left_pos == active.end() || right_pos == active.end()) {
            throw std::logic_error("invalid disconnected-component parenthesization state");
        }
        auto i = static_cast<std::size_t>(left_pos - active.begin());
        auto j = static_cast<std::size_t>(right_pos - active.begin());
        if (i > j) {
            std::swap(i, j);
        }
        path.emplace_back(i, j);
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
        accumulated |= next;
        active.push_back(accumulated);
    }
    return {make_candidate(path, node, spaces, sizes, model, cost_factors)};
}

std::vector<ParenthesizationCandidate> dynamic_programming(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const ScalingLimits& limits, const CostModel& model,
    const CostFactors& cost_factors) {
    std::vector<ParenthesizationCandidate> tensor_result;
    if (dynamic_programming_bitmask(node, spaces, sizes, limits, model,
                                    cost_factors, tensor_result)) {
        return tensor_result;
    }
    // Inputs beyond the mask limits use general Axes sets.
    return dynamic_programming_generic(node, checker, spaces, sizes, model,
                                       cost_factors);
}

}  // namespace

std::vector<ParenthesizationCandidate> parenthesization_candidates(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits, std::size_t max_candidates,
    std::size_t exhaustive_limit, const CostModel& model,
    const CostFactors& cost_factors) {
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
        return dynamic_programming(node, checker, spaces, sizes, limits,
                                   model, cost_factors);
    }
    if (max_candidates == 0) {
        return {};
    }
    return exhaustive(node, checker, spaces, sizes, model, cost_factors, max_candidates);
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
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(first));
        active.push_back(std::move(result));
    }
    if (active.size() != 1) {
        throw std::invalid_argument("candidate path must contract all Mul operands");
    }
    return active.front();
}

}  // namespace albert
