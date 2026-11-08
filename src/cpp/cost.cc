#include "cost.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace albert {
namespace {

using AxisSet = std::set<std::string>;
using IndexSpaceMap = std::unordered_map<std::string, std::string>;

struct Totals {
    double multiplications = 0.0;
    double additions = 0.0;
    double reads = 0.0;
    double writes = 0.0;
};

struct Summary {
    AxisSet axes;
    double elements = 1.0;
    Totals totals;
};

IndexSpaceMap make_index_space_map(const IndexSpaces& spaces) {
    IndexSpaceMap result;
    for (const auto& [space, indices] : spaces) {
        for (const auto& index : indices) {
            const auto [found, inserted] = result.emplace(index, space);
            if (!inserted && found->second != space) {
                throw std::invalid_argument("an index cannot belong to multiple spaces");
            }
        }
    }
    return result;
}

AxisSet union_axes(const AxisSet& left, const AxisSet& right) {
    AxisSet result;
    std::set_union(left.begin(), left.end(), right.begin(), right.end(),
                   std::inserter(result, result.end()));
    return result;
}

AxisSet intersect_axes(const AxisSet& left, const AxisSet& right) {
    AxisSet result;
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::inserter(result, result.end()));
    return result;
}

AxisSet symmetric_difference_axes(const AxisSet& left, const AxisSet& right) {
    AxisSet result;
    std::set_symmetric_difference(left.begin(), left.end(), right.begin(), right.end(),
                                  std::inserter(result, result.end()));
    return result;
}

class Estimator {
public:
    Estimator(const IndexSpaceMap& index_spaces, const SpaceSizes& sizes)
        : index_spaces_(index_spaces), sizes_(sizes) {}

    double size(const AxisSet& axes) const {
        double result = 1.0;
        for (const auto& index : axes) {
            const auto space = index_spaces_.find(index);
            if (space == index_spaces_.end()) {
                throw std::invalid_argument("index " + index + " does not belong to a space");
            }
            const auto dimension = sizes_.find(space->second);
            if (dimension == sizes_.end()) {
                throw std::invalid_argument("space " + space->second + " has no declared size");
            }
            if (dimension->second == 0) {
                throw std::invalid_argument("space sizes must be positive");
            }
            result *= static_cast<double>(dimension->second);
            if (!std::isfinite(result)) {
                throw std::overflow_error("estimated tensor size exceeds numeric range");
            }
        }
        return result;
    }

    Summary visit(const NodePtr& node) const {
        if (const auto* scalar = dynamic_cast<const Scalar*>(node.get())) {
            static_cast<void>(scalar);
            return {};
        }
        if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
            Summary result;
            AxisSet all_axes;
            std::unordered_map<std::string, std::size_t> counts;
            for (const auto& index : tensor->indices()) {
                all_axes.insert(index);
                ++counts[index];
            }
            for (const auto& [index, count] : counts) {
                if (count == 1) {
                    result.axes.insert(index);
                } else if (count > 2) {
                    throw std::invalid_argument("Einstein index occurs more than twice");
                }
            }
            result.elements = size(result.axes);
            if (all_axes.size() != result.axes.size()) {
                const auto input_elements = size(all_axes);
                const auto reduction = input_elements / result.elements;
                result.totals.additions = result.elements * (reduction - 1.0);
                result.totals.reads = input_elements;
                result.totals.writes = result.elements;
            }
            return result;
        }
        if (node->kind() == NodeKind::Add) {
            return visit_add(node->children());
        }
        return visit_mul(node->children());
    }

private:
    Summary visit_add(const Children& children) const {
        if (children.empty()) {
            return {};
        }

        auto result = visit(children.front());
        for (std::size_t i = 1; i < children.size(); ++i) {
            auto right = visit(children[i]);
            if (result.axes != right.axes) {
                throw std::invalid_argument("addition operands must have the same external indices");
            }
            add_totals(result.totals, right.totals);
            result.totals.additions += result.elements;
            result.totals.reads += result.elements + right.elements;
            result.totals.writes += result.elements;
        }
        return result;
    }

    Summary visit_mul(const Children& children) const {
        if (children.empty()) {
            return {};
        }

        auto result = visit(children.front());
        for (std::size_t i = 1; i < children.size(); ++i) {
            auto right = visit(children[i]);
            add_totals(result.totals, right.totals);

            const auto joined_axes = union_axes(result.axes, right.axes);
            const auto summed_axes = intersect_axes(result.axes, right.axes);
            const auto output_axes = symmetric_difference_axes(result.axes, right.axes);
            const auto work = size(joined_axes);
            const auto reduction = size(summed_axes);
            const auto output_size = size(output_axes);

            result.totals.multiplications += work;
            result.totals.additions += output_size * (reduction - 1.0);
            result.totals.reads += result.elements + right.elements;
            result.totals.writes += output_size;
            result.axes = output_axes;
            result.elements = output_size;
        }
        return result;
    }

    static void add_totals(Totals& target, const Totals& source) {
        target.multiplications += source.multiplications;
        target.additions += source.additions;
        target.reads += source.reads;
        target.writes += source.writes;
    }

    const IndexSpaceMap& index_spaces_;
    const SpaceSizes& sizes_;
};

void validate_model(const CostModel& model) {
    const double weights[] = {model.multiply, model.add, model.read, model.write};
    for (const auto weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("cost model weights must be finite and nonnegative");
        }
    }
}

}  // namespace

CostEstimate estimate_cost(const NodePtr& node, const IndexSpaces& spaces,
                           const SpaceSizes& sizes, const CostModel& model) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
    validate_model(model);
    for (const auto& [space, indices] : spaces) {
        static_cast<void>(indices);
        if (sizes.find(space) == sizes.end()) {
            throw std::invalid_argument("space " + space + " has no declared size");
        }
    }
    for (const auto& [space, dimension] : sizes) {
        static_cast<void>(space);
        if (dimension == 0) {
            throw std::invalid_argument("space sizes must be positive");
        }
    }
    const auto index_spaces = make_index_space_map(spaces);
    const auto totals = Estimator(index_spaces, sizes).visit(node).totals;
    const auto total = totals.multiplications * model.multiply +
                       totals.additions * model.add + totals.reads * model.read +
                       totals.writes * model.write;
    if (!std::isfinite(total)) {
        throw std::overflow_error("estimated cost exceeds numeric range");
    }
    return {
        totals.multiplications,
        totals.additions,
        totals.reads,
        totals.writes,
        total,
    };
}

}  // namespace albert
