#include "parenthesize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace albert {
namespace {

using Axes = std::set<std::string>;

int worker_count(std::size_t work_items) noexcept {
#ifdef _OPENMP
    constexpr std::size_t large_batch = 16'384;
    const auto maximum = omp_get_max_threads();
    return work_items >= large_batch ? maximum : std::min(2, maximum);
#else
    static_cast<void>(work_items);
    return 1;
#endif
}

template <typename Callback>
void parallel_for(std::size_t count, std::size_t threshold, Callback&& callback) {
    const auto workers = worker_count(count);
    if (count < threshold || workers == 1) {
        for (std::size_t index = 0; index < count; ++index) {
            callback(index);
        }
        return;
    }
    std::vector<std::exception_ptr> errors(count);
#pragma omp parallel for schedule(dynamic) num_threads(workers)
    for (std::ptrdiff_t item = 0; item < static_cast<std::ptrdiff_t>(count); ++item) {
        const auto index = static_cast<std::size_t>(item);
        try {
            callback(index);
        } catch (...) {
            errors[index] = std::current_exception();
        }
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

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

bool candidate_cost_less(const ParenthesizationCandidate& lhs,
                         const ParenthesizationCandidate& rhs) {
    return lhs.cost.total < rhs.cost.total;
}

void retain_best(std::vector<ParenthesizationCandidate>& candidates,
                 ParenthesizationCandidate candidate, std::size_t maximum) {
    candidates.push_back(std::move(candidate));
    std::stable_sort(candidates.begin(), candidates.end(), candidate_cost_less);
    if (candidates.size() > maximum) {
        candidates.pop_back();
    }
}

std::vector<ParenthesizationCandidate> exhaustive(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model,
    const CostFactors& cost_factors, std::size_t maximum) {
    std::vector<Active> active;
    active.reserve(node.children().size());
    for (const auto& child : node.children()) {
        active.push_back({child, get_axes(child)});
    }

    if (active.size() == 1) {
        return {make_candidate({}, node, spaces, sizes, model, cost_factors)};
    }

    std::vector<std::pair<std::size_t, std::size_t>> first_choices;
    for (std::size_t i = 0; i < active.size(); ++i) {
        for (std::size_t j = i + 1; j < active.size(); ++j) {
            const auto operation = unite(active[i].axes, active[j].axes);
            const auto output = xor_axes(active[i].axes, active[j].axes);
            if (checker.allows(operation, output)) {
                first_choices.emplace_back(i, j);
            }
        }
    }
    std::vector<std::vector<ParenthesizationCandidate>> branch_results(first_choices.size());
    parallel_for(first_choices.size(), 8, [&](std::size_t index) {
        const auto [i, j] = first_choices[index];
        const auto output = xor_axes(active[i].axes, active[j].axes);
        auto next = active;
        next.erase(next.begin() + static_cast<std::ptrdiff_t>(j));
        next.erase(next.begin() + static_cast<std::ptrdiff_t>(i));
        next.push_back({std::make_shared<Mul>(Children{active[i].node, active[j].node}),
                        output});
        std::vector<Contraction> path{{i, j}};
        auto& best = branch_results[index];
        std::function<void(const std::vector<Active>&)> visit;
        visit = [&](const std::vector<Active>& current) {
            if (current.size() == 1) {
                retain_best(best, make_candidate(path, node, spaces, sizes, model,
                                                 cost_factors), maximum);
                return;
            }
            for (std::size_t lhs = 0; lhs < current.size(); ++lhs) {
                for (std::size_t rhs = lhs + 1; rhs < current.size(); ++rhs) {
                    const auto child_operation = unite(current[lhs].axes,
                                                       current[rhs].axes);
                    const auto child_output = xor_axes(current[lhs].axes,
                                                       current[rhs].axes);
                    if (!checker.allows(child_operation, child_output)) {
                        continue;
                    }
                    auto child_state = current;
                    child_state.erase(child_state.begin() +
                                      static_cast<std::ptrdiff_t>(rhs));
                    child_state.erase(child_state.begin() +
                                      static_cast<std::ptrdiff_t>(lhs));
                    child_state.push_back({
                        std::make_shared<Mul>(Children{current[lhs].node,
                                                       current[rhs].node}),
                        child_output});
                    path.emplace_back(lhs, rhs);
                    visit(child_state);
                    path.pop_back();
                }
            }
        };
        visit(next);
    });

    std::vector<ParenthesizationCandidate> best;
    for (std::size_t index = 0; index < branch_results.size(); ++index) {
        for (auto& candidate : branch_results[index]) {
            best.push_back(std::move(candidate));
        }
    }
    std::stable_sort(best.begin(), best.end(), candidate_cost_less);
    if (best.size() > maximum) {
        best.resize(maximum);
    }
    return best;
}

struct DpEntry {
    double score = std::numeric_limits<double>::infinity();
    std::uint64_t left = 0;
    std::uint64_t right = 0;
};

struct DpSplit {
    std::uint64_t left;
    std::uint64_t right;
    double operation_size = std::numeric_limits<double>::infinity();
};

struct BitmaskDpEntry {
    double score = std::numeric_limits<double>::infinity();
    std::uint64_t left = 0;
    std::uint64_t right = 0;
    bool computed = false;
};

template <typename Entry>
void update_best_split(Entry& entry, const DpSplit& split,
                       double left_score, double right_score) {
    if (!std::isfinite(left_score) || !std::isfinite(right_score)) {
        return;
    }
    const auto score = left_score + right_score + split.operation_size;
    if (score < entry.score) {
        entry.score = score;
        entry.left = split.left;
        entry.right = split.right;
    }
}

template <typename EntryLookup>
void append_dp_path(std::uint64_t root, std::vector<std::uint64_t>& active,
                    std::vector<Contraction>& path, EntryLookup&& lookup) {
    const auto emit = [&](const auto& self, std::uint64_t mask) -> void {
        const auto* entry = lookup(mask);
        if (entry == nullptr || entry->left == 0) {
            return;
        }
        self(self, entry->left);
        self(self, entry->right);
        const auto left = std::find(active.begin(), active.end(), entry->left);
        const auto right = std::find(active.begin(), active.end(), entry->right);
        if (left == active.end() || right == active.end()) {
            throw std::logic_error("invalid DP parenthesization state");
        }
        auto i = static_cast<std::size_t>(left - active.begin());
        auto j = static_cast<std::size_t>(right - active.begin());
        if (i > j) {
            std::swap(i, j);
        }
        path.emplace_back(i, j);
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
        active.push_back(entry->left | entry->right);
    };
    emit(emit, root);
}

struct MaskCap {
    std::uint64_t axes = 0;
    std::size_t maximum = 0;
};

double allowed_operation_size(const Axes& left, const Axes& right,
                              const LimitsChecker& checker) {
    if (!shares_axis(left, right)) {
        return std::numeric_limits<double>::infinity();
    }
    const auto operation = unite(left, right);
    if (!checker.allows(operation, xor_axes(left, right))) {
        return std::numeric_limits<double>::infinity();
    }
    return checker.size(operation);
}

std::size_t popcount(std::uint64_t value) noexcept {
    value -= (value >> 1) & 0x5555555555555555ULL;
    value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<std::size_t>((value * 0x0101010101010101ULL) >> 56);
}

template <typename Positions, typename Callback>
void for_each_bitmask_partition(std::uint64_t mask, const Positions& positions,
                                std::size_t count, Callback&& callback) {
    for (std::size_t left_count = 1; left_count <= count / 2; ++left_count) {
        std::array<std::size_t, 20> combination{};
        std::uint64_t left = 0;
        for (std::size_t i = 0; i < left_count; ++i) {
            combination[i] = i;
            left |= std::uint64_t{1} << positions[i];
        }
        for (;;) {
            const auto right = mask ^ left;
            if (left_count * 2 != count || left < right) {
                callback(left, right);
            }
            std::size_t position = left_count;
            bool advanced = false;
            while (position > 0) {
                --position;
                if (combination[position] < count - left_count + position) {
                    for (std::size_t j = position; j < left_count; ++j) {
                        left &= ~(std::uint64_t{1} << positions[combination[j]]);
                    }
                    ++combination[position];
                    for (std::size_t j = position + 1; j < left_count; ++j) {
                        combination[j] = combination[j - 1] + 1;
                    }
                    for (std::size_t j = position; j < left_count; ++j) {
                        left |= std::uint64_t{1} << positions[combination[j]];
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
}

template <typename Callback>
void for_each_partition(std::uint64_t mask, const std::vector<std::size_t>& positions,
                        Callback&& callback) {
    const auto count = positions.size();
    for (std::size_t left_count = 1; left_count <= count / 2; ++left_count) {
        std::vector<std::size_t> combination(left_count);
        for (std::size_t i = 0; i < left_count; ++i) {
            combination[i] = i;
        }
        for (;;) {
            std::uint64_t left = 0;
            for (const auto position : combination) {
                left |= std::uint64_t{1} << positions[position];
            }
            if (left_count * 2 != count || left < (mask ^ left)) {
                callback(left, mask ^ left);
            }
            std::size_t position = left_count;
            bool advanced = false;
            while (position > 0) {
                --position;
                if (combination[position] < count - left_count + position) {
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
}

template <typename SharesAxes>
std::vector<std::uint64_t> connected_components(std::size_t count,
                                                SharesAxes&& shares_axes) {
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
                if (shares_axes(component, *it)) {
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
    return components;
}

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
            if (popcount(axes & cap.axes) > cap.maximum) {
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
    const auto contraction_size = [&](std::uint64_t left, std::uint64_t right) {
        if ((left & right) == 0 || !under_caps(left | right, time_caps)) {
            return std::numeric_limits<double>::infinity();
        }
        return axes_size(left | right);
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
    std::unordered_set<std::uint64_t> dp_roots;
    std::vector<std::uint64_t> subset_axes_memo(state_count);
    std::vector<bool> subset_axes_computed(state_count, false);
    const auto subset_axes = [&](const auto& self, std::uint64_t mask) -> std::uint64_t {
        if (subset_axes_computed[mask]) {
            return subset_axes_memo[mask];
        }
        const auto first_bit = mask & (~mask + 1);
        std::uint64_t axes;
        if (mask == first_bit) {
            axes = leaf_axes[static_cast<std::size_t>(__builtin_ctzll(mask))];
        } else {
            axes = leaf_axes[static_cast<std::size_t>(__builtin_ctzll(first_bit))] ^
                   self(self, mask ^ first_bit);
        }
        subset_axes_computed[mask] = true;
        subset_axes_memo[mask] = axes;
        return axes;
    };
    const auto solve = [&](const auto& self, std::uint64_t mask) -> const BitmaskDpEntry& {
        auto& entry = memo[mask];
        if (entry.computed) {
            return entry;
        }
        entry.computed = true;
        if ((mask & (mask - 1)) == 0) {
            entry.score = 0.0;
            return entry;
        }
        if (!under_caps(subset_axes(subset_axes, mask), memory_caps)) {
            return entry;
        }
        std::array<std::size_t, 20> positions{};
        std::size_t subset_count = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (mask & (std::uint64_t{1} << i)) {
                positions[subset_count++] = i;
            }
        }
        if (dp_roots.find(mask) == dp_roots.end() || subset_count < 8) {
            for_each_bitmask_partition(mask, positions, subset_count,
                                       [&](std::uint64_t left, std::uint64_t right) {
                const auto left_axes = subset_axes(subset_axes, left);
                const auto right_axes = subset_axes(subset_axes, right);
                const auto cost = contraction_size(left_axes, right_axes);
                if (!std::isfinite(cost)) {
                    return;
                }
                const auto& lhs = self(self, left);
                const auto& rhs = self(self, right);
                update_best_split(entry, DpSplit{left, right, cost}, lhs.score,
                                  rhs.score);
            });
            return entry;
        }
        std::vector<DpSplit> splits;
        for_each_bitmask_partition(mask, positions, subset_count,
                                   [&](std::uint64_t left, std::uint64_t right) {
            splits.push_back({left, right});
        });
        for (const auto& split : splits) {
            static_cast<void>(subset_axes(subset_axes, split.left));
            static_cast<void>(subset_axes(subset_axes, split.right));
        }
#pragma omp parallel for schedule(static) num_threads(worker_count(splits.size())) if (splits.size() >= 64)
        for (std::ptrdiff_t split_index = 0;
             split_index < static_cast<std::ptrdiff_t>(splits.size()); ++split_index) {
            auto& split = splits[static_cast<std::size_t>(split_index)];
            const auto left_axes = subset_axes_memo[split.left];
            const auto right_axes = subset_axes_memo[split.right];
            split.operation_size = contraction_size(left_axes, right_axes);
        }
        for (const auto& split : splits) {
            if (!std::isfinite(split.operation_size)) {
                continue;
            }
            const auto& lhs = self(self, split.left);
            const auto& rhs = self(self, split.right);
            update_best_split(entry, split, lhs.score, rhs.score);
        }
        return entry;
    };

    std::vector<std::uint64_t> active;
    active.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        active.push_back(std::uint64_t{1} << i);
    }
    std::vector<Contraction> path;

    auto components = connected_components(count, [&](auto lhs, auto rhs) {
        return (subset_axes(subset_axes, lhs) & subset_axes(subset_axes, rhs)) != 0;
    });
    for (const auto component : components) {
        dp_roots.insert(component);
    }
    for (const auto component : components) {
        if (!std::isfinite(solve(solve, component).score)) {
            return true;
        }
        append_dp_path(component, active, path, [&](std::uint64_t mask) {
            return &solve(solve, mask);
        });
    }
    std::stable_sort(components.begin(), components.end(), [&](auto lhs, auto rhs) {
        return axes_size(subset_axes(subset_axes, lhs)) <
               axes_size(subset_axes(subset_axes, rhs));
    });
    auto accumulated = components.front();
    for (std::size_t k = 1; k < components.size(); ++k) {
        const auto next = components[k];
        const auto operation = subset_axes(subset_axes, accumulated) |
                               subset_axes(subset_axes, next);
        const auto output = subset_axes(subset_axes, accumulated) ^
                            subset_axes(subset_axes, next);
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
    std::unordered_set<std::uint64_t> dp_roots;
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
        if (dp_roots.find(mask) == dp_roots.end() || positions.size() < 8) {
            for_each_partition(mask, positions,
                               [&](std::uint64_t left, std::uint64_t right) {
                const auto& left_axes = subset_axes(left);
                const auto& right_axes = subset_axes(right);
                const auto cost = allowed_operation_size(left_axes, right_axes, checker);
                if (!std::isfinite(cost)) {
                    return;
                }
                const auto lhs_score = solve(left).score;
                const auto rhs_score = solve(right).score;
                update_best_split(entry, DpSplit{left, right, cost}, lhs_score,
                                  rhs_score);
            });
            return memo.emplace(mask, std::move(entry)).first->second;
        }
        std::vector<DpSplit> splits;
        for_each_partition(mask, positions,
                           [&](std::uint64_t left, std::uint64_t right) {
            splits.push_back({left, right});
        });
        for (const auto& split : splits) {
            static_cast<void>(subset_axes(split.left));
            static_cast<void>(subset_axes(split.right));
        }
        const auto& axes_cache = subset_axes_memo;
        parallel_for(splits.size(), 64, [&](std::size_t index) {
            auto& split = splits[index];
            const auto& left_axes = axes_cache.at(split.left);
            const auto& right_axes = axes_cache.at(split.right);
            split.operation_size = allowed_operation_size(left_axes, right_axes, checker);
        });
        for (const auto& split : splits) {
            if (!std::isfinite(split.operation_size)) {
                continue;
            }
            const auto lhs_score = solve(split.left).score;
            const auto rhs_score = solve(split.right).score;
            update_best_split(entry, split, lhs_score, rhs_score);
        }
        return memo.emplace(mask, std::move(entry)).first->second;
    };
    std::vector<Contraction> path;
    std::vector<std::uint64_t> active;
    for (std::size_t i = 0; i < count; ++i) {
        active.push_back(std::uint64_t{1} << i);
    }
    auto components = connected_components(count, [&](auto lhs, auto rhs) {
        return shares_axis(subset_axes(lhs), subset_axes(rhs));
    });
    for (const auto component : components) {
        dp_roots.insert(component);
    }
    for (const auto component : components) {
        const auto& root = solve(component);
        if (!std::isfinite(root.score)) {
            return {};
        }
        append_dp_path(component, active, path, [&](std::uint64_t mask) {
            const auto found = memo.find(mask);
            return found == memo.end() ? nullptr : &found->second;
        });
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

std::vector<ParenthesizationCandidate> branch_search(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model,
    const CostFactors& cost_factors, std::size_t width) {
    struct State {
        std::vector<Active> active;
        std::vector<Contraction> path;
        double lower_bound = 0.0;
    };
    std::vector<Active> initial;
    for (const auto& child : node.children()) {
        initial.push_back({child, get_axes(child)});
    }
    double best_cost = std::numeric_limits<double>::infinity();
    std::optional<ParenthesizationCandidate> best;
    const auto node_budget = width * node.children().size();
    std::size_t explored = 0;
    std::function<void(const State&)> visit = [&](const State& state) {
        if (explored >= node_budget) {
            return;
        }
        if (state.active.size() == 1) {
            auto candidate = make_candidate(state.path, node, spaces, sizes, model, cost_factors);
            if (candidate.cost.total < best_cost) {
                best_cost = candidate.cost.total;
                best = std::move(candidate);
            }
            return;
        }
        if (state.lower_bound >= best_cost) {
            return;
        }
        ++explored;
        struct Choice {
            std::size_t i;
            std::size_t j;
            Axes operation;
            Axes output;
            double score;
        };
        std::vector<Choice> choices;
        for (std::size_t i = 0; i < state.active.size(); ++i) {
            for (std::size_t j = i + 1; j < state.active.size(); ++j) {
                auto operation = unite(state.active[i].axes, state.active[j].axes);
                auto output = xor_axes(state.active[i].axes, state.active[j].axes);
                if (checker.allows(operation, output)) {
                    choices.push_back({i, j, std::move(operation), std::move(output), 0.0});
                    choices.back().score = checker.size(choices.back().operation);
                }
            }
        }
        std::stable_sort(choices.begin(), choices.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.score < rhs.score;
        });
        if (choices.size() > width) {
            choices.resize(width);
        }
        for (const auto& choice : choices) {
            State next = state;
            next.active.erase(next.active.begin() + static_cast<std::ptrdiff_t>(choice.j));
            next.active.erase(next.active.begin() + static_cast<std::ptrdiff_t>(choice.i));
            next.active.push_back({
                std::make_shared<Mul>(Children{state.active[choice.i].node,
                                               state.active[choice.j].node}),
                choice.output});
            next.path.emplace_back(choice.i, choice.j);
            next.lower_bound += choice.score * model.multiply;
            visit(next);
        }
    };
    visit(State{std::move(initial), {}, 0.0});
    return best ? std::vector<ParenthesizationCandidate>{std::move(*best)}
                : std::vector<ParenthesizationCandidate>{};
}

std::vector<ParenthesizationCandidate> greedy_search(
    const Mul& node, const LimitsChecker& checker, const IndexSpaces& spaces,
    const SpaceSizes& sizes, const CostModel& model,
    const CostFactors& cost_factors, std::size_t restarts) {
    std::vector<std::vector<ParenthesizationCandidate>> restart_results(restarts);
    parallel_for(restarts, 4, [&](std::size_t restart) {
        std::vector<Active> active;
        for (const auto& child : node.children()) {
            active.push_back({child, get_axes(child)});
        }
        std::vector<Contraction> path;
        while (active.size() > 1) {
            struct Choice { std::size_t i; std::size_t j; double score; Axes output; };
            std::vector<Choice> choices;
            for (std::size_t i = 0; i < active.size(); ++i) {
                for (std::size_t j = i + 1; j < active.size(); ++j) {
                    const auto operation = unite(active[i].axes, active[j].axes);
                    auto output = xor_axes(active[i].axes, active[j].axes);
                    if (checker.allows(operation, output)) {
                        choices.push_back({i, j, checker.size(operation),
                                           std::move(output)});
                    }
                }
            }
            if (choices.empty()) {
                path.clear();
                break;
            }
            std::stable_sort(choices.begin(), choices.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.score < rhs.score;
            });
            const auto choice_index = std::min(
                restart % std::min<std::size_t>(choices.size(), 3), choices.size() - 1);
            const auto choice = choices[choice_index];
            path.emplace_back(choice.i, choice.j);
            auto merged = std::make_shared<Mul>(Children{active[choice.i].node,
                                                         active[choice.j].node});
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(choice.j));
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(choice.i));
            active.push_back({std::move(merged), choice.output});
        }
        if (active.size() == 1 && path.size() + 1 == node.children().size()) {
            restart_results[restart].push_back(
                make_candidate(path, node, spaces, sizes, model, cost_factors));
        }
    });
    std::vector<ParenthesizationCandidate> candidates;
    for (std::size_t restart = 0; restart < restarts; ++restart) {
        for (auto& candidate : restart_results[restart]) {
            if (std::none_of(candidates.begin(), candidates.end(), [&](const auto& existing) {
                    return existing.path == candidate.path;
                })) {
                candidates.push_back(std::move(candidate));
            }
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), candidate_cost_less);
    return candidates;
}

}  // namespace

const PathRequest& PathRequests::for_size(std::size_t operand_count) const {
    if (requests.size() != bounds.size() + 1 ||
        !std::is_sorted(bounds.begin(), bounds.end()) ||
        std::adjacent_find(bounds.begin(), bounds.end()) != bounds.end()) {
        throw std::invalid_argument("path request bounds and requests are inconsistent");
    }
    const auto found = std::lower_bound(bounds.begin(), bounds.end(), operand_count);
    return requests[static_cast<std::size_t>(found - bounds.begin())];
}

std::vector<ParenthesizationCandidate> parenthesization_candidates(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits, std::size_t max_candidates,
    std::size_t exhaustive_limit, const CostModel& model,
    const CostFactors& cost_factors) {
    PathRequest request;
    request.dynamic = node.children().size() > exhaustive_limit;
    request.exhaustive = request.dynamic ? 0 : max_candidates;
    return parenthesization_candidates(node, spaces, sizes, limits, request,
                                       max_candidates, model, cost_factors);
}

std::vector<ParenthesizationCandidate> parenthesization_candidates(
    const Mul& node, const IndexSpaces& spaces, const SpaceSizes& sizes,
    const ScalingLimits& limits, const PathRequest& request,
    std::size_t max_candidates, const CostModel& model,
    const CostFactors& cost_factors) {
    if (node.children().empty()) {
        throw std::invalid_argument("cannot parenthesize an empty Mul");
    }
    if (max_candidates == 0 && !request.dynamic) {
        return {};
    }
    LimitsChecker checker(spaces, sizes, limits);
    for (const auto& child : node.children()) {
        if (!checker.memory_allows_tree(child) || !checker.time_allows_tree(child)) {
            return {};
        }
    }
    if (max_candidates == 0) {
        return dynamic_programming(node, checker, spaces, sizes, limits,
                                   model, cost_factors);
    }
    std::vector<ParenthesizationCandidate> result;
    auto append_unique = [&](std::vector<ParenthesizationCandidate> new_candidates) {
        for (auto& candidate : new_candidates) {
            if (std::none_of(result.begin(), result.end(), [&](const auto& existing) {
                    return existing.path == candidate.path;
                })) {
                result.push_back(std::move(candidate));
            }
        }
    };
    if (request.dynamic) {
        append_unique(dynamic_programming(node, checker, spaces, sizes, limits,
                                          model, cost_factors));
    }
    if (request.exhaustive > 0) {
        append_unique(exhaustive(node, checker, spaces, sizes, model, cost_factors,
                                 request.exhaustive));
    }
    if (request.branch > 0) {
        append_unique(branch_search(node, checker, spaces, sizes, model,
                                    cost_factors, request.branch));
    }
    if (request.greedy > 0) {
        append_unique(greedy_search(node, checker, spaces, sizes, model,
                                   cost_factors, request.greedy));
    }
    std::stable_sort(result.begin(), result.end(), candidate_cost_less);
    if (result.size() > max_candidates && max_candidates > 0) {
        result.resize(max_candidates);
    }
    return result;
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
