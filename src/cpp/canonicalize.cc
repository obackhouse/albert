#include "canonicalize.h"

#include "algebra.h"
#include "search.h"
#include "simplify.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace albert {
namespace {

using IndexMap = std::unordered_map<std::string, std::string>;
using SpaceMap = std::unordered_map<std::string, std::string>;
using Symmetry = std::pair<std::vector<std::size_t>, int>;

struct Form {
    NodePtr node;
    int phase = 1;
};

struct LabelGroup {
    std::string space;
    std::vector<std::string> sources;
    std::vector<std::string> targets;
};

struct ExternalMappingSet {
    std::vector<IndexMap> mappings;
    std::unordered_map<std::string, std::string> spaces;
};

bool node_less(const NodePtr& lhs, const NodePtr& rhs);

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0;
    if (value != 0.0) {
        std::memcpy(&bits, &value, sizeof(bits));
    }
    return bits;
}

bool vector_less(const Children& lhs, const Children& rhs) {
    const auto common = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < common; ++i) {
        if (node_less(lhs[i], rhs[i])) {
            return true;
        }
        if (node_less(rhs[i], lhs[i])) {
            return false;
        }
    }
    return lhs.size() < rhs.size();
}

bool node_less(const NodePtr& lhs, const NodePtr& rhs) {
    if (canonical_key(lhs) == canonical_key(rhs)) {
        return false;
    }
    if (lhs->kind() != rhs->kind()) {
        return lhs->kind() < rhs->kind();
    }
    if (const auto* left = dynamic_cast<const Scalar*>(lhs.get())) {
        const auto* right = dynamic_cast<const Scalar*>(rhs.get());
        return double_bits(left->value()) < double_bits(right->value());
    }
    if (const auto* left = dynamic_cast<const Tensor*>(lhs.get())) {
        const auto* right = dynamic_cast<const Tensor*>(rhs.get());
        if (left->symbol() != right->symbol()) {
            return left->symbol() < right->symbol();
        }
        return left->indices() < right->indices();
    }
    return vector_less(lhs->children(), rhs->children());
}

void collect_indices(const NodePtr& node, std::set<std::string>& indices) {
    for (const auto& current : albert::search(
             node, [](const NodePtr&) { return true; })) {
        if (const auto* tensor = dynamic_cast<const Tensor*>(current.get())) {
            indices.insert(tensor->indices().begin(), tensor->indices().end());
        }
    }
}

std::vector<std::string> unique_indices(const std::vector<std::string>& indices) {
    std::vector<std::string> result = indices;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

SpaceMap make_space_map(const CanonicalizationMetadata& metadata) {
    SpaceMap result;
    for (const auto& [space, indices] : metadata.spaces) {
        for (const auto& index : indices) {
            const auto [entry, inserted] = result.emplace(index, space);
            if (!inserted && entry->second != space) {
                throw std::invalid_argument("an index cannot belong to multiple spaces");
            }
        }
    }
    return result;
}

std::string index_space(const std::string& index, const SpaceMap& spaces) {
    const auto found = spaces.find(index);
    if (found != spaces.end()) {
        return found->second;
    }
    constexpr const char* prefix = "__internal_";
    constexpr std::size_t prefix_size = 11;
    if (index.compare(0, prefix_size, prefix) == 0 && index.size() > prefix_size + 3 &&
        index.compare(index.size() - 2, 2, "__") == 0) {
        const auto body = index.substr(prefix_size, index.size() - prefix_size - 2);
        const auto separator = body.rfind('_');
        if (separator != std::string::npos && separator != 0 &&
            separator + 1 < body.size() &&
            std::all_of(body.begin() + static_cast<std::ptrdiff_t>(separator + 1),
                        body.end(), [](unsigned char value) { return std::isdigit(value); })) {
            return body.substr(0, separator);
        }
    }
    return {};
}

std::vector<std::string> space_labels(const std::string& space,
                                     const CanonicalizationMetadata& metadata) {
    const auto found = metadata.spaces.find(space);
    if (found == metadata.spaces.end()) {
        return {};
    }
    auto labels = found->second;
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    return labels;
}

void enumerate_group_mappings(const std::vector<LabelGroup>& groups,
                              std::size_t group_position, IndexMap& current,
                              std::vector<IndexMap>& results);

ExternalMappingSet external_mappings(
    const NodePtr& node, const SpaceMap& spaces,
    const std::set<std::string>& used_indices) {
    std::map<std::string, std::vector<std::string>> source_groups;
    for (const auto& index : node->external_indices()) {
        auto& sources = source_groups[index_space(index, spaces)];
        if (std::find(sources.begin(), sources.end(), index) == sources.end()) {
            sources.push_back(index);
        }
    }

    IndexMap mapping;
    std::unordered_map<std::string, std::string> placeholder_spaces;
    std::size_t group_number = 0;
    for (auto& [space, sources] : source_groups) {
        for (std::size_t i = 0; i < sources.size(); ++i) {
            auto placeholder = "__external_" + std::to_string(group_number) + "_" +
                               std::to_string(i);
            while (used_indices.find(placeholder) != used_indices.end()) {
                placeholder.push_back('_');
            }
            mapping.emplace(sources[i], placeholder);
            placeholder_spaces.emplace(placeholder, space);
        }
        ++group_number;
    }
    return {{std::move(mapping)}, std::move(placeholder_spaces)};
}

std::vector<IndexMap> final_external_mappings(
    const NodePtr& node, const std::unordered_map<std::string, std::string>& placeholder_spaces,
    const CanonicalizationMetadata& metadata) {
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto& index : node->external_indices()) {
        const auto found = placeholder_spaces.find(index);
        if (found != placeholder_spaces.end()) {
            groups[found->second].push_back(index);
        }
    }

    std::set<std::string> used;
    collect_indices(node, used);
    for (const auto& [placeholder, space] : placeholder_spaces) {
        static_cast<void>(space);
        used.erase(placeholder);
    }

    IndexMap mapping;
    std::size_t group_number = 0;
    for (auto& [space, sources] : groups) {
        LabelGroup group{space, sources, {}};
        for (const auto& label : space_labels(space, metadata)) {
            if (used.find(label) == used.end()) {
                group.targets.push_back(label);
            }
            if (group.targets.size() == sources.size()) {
                break;
            }
        }
        for (std::size_t i = group.targets.size(); i < sources.size(); ++i) {
            auto label = "__external_" + std::to_string(group_number) + "_" +
                         std::to_string(i);
            while (used.find(label) != used.end()) {
                label.push_back('_');
            }
            group.targets.push_back(std::move(label));
        }
        for (std::size_t i = 0; i < sources.size(); ++i) {
            mapping.emplace(sources[i], group.targets[i]);
        }
        ++group_number;
    }
    return {std::move(mapping)};
}

NodePtr map_external_indices(const NodePtr& node, const IndexMap& mapping) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        const auto external = tensor->external_indices();
        const std::unordered_set<std::string> external_set(external.begin(), external.end());
        auto indices = tensor->indices();
        bool changed = false;
        for (auto& index : indices) {
            const auto found = mapping.find(index);
            if (external_set.find(index) != external_set.end() && found != mapping.end() &&
                found->second != index) {
                index = found->second;
                changed = true;
            }
        }
        return changed ? std::make_shared<Tensor>(tensor->symbol(), std::move(indices)) : node;
    }
    if (node->kind() == NodeKind::Scalar) {
        return node;
    }

    Children children;
    children.reserve(node->children().size());
    bool changed = false;
    for (const auto& child : node->children()) {
        IndexMap child_mapping;
        const auto child_external = child->external_indices();
        for (const auto& index : child_external) {
            if (const auto found = mapping.find(index); found != mapping.end()) {
                child_mapping.emplace(index, found->second);
            }
        }
        auto mapped = map_external_indices(child, child_mapping);
        changed = changed || mapped != child;
        children.push_back(std::move(mapped));
    }
    if (!changed) {
        return node;
    }
    if (node->kind() == NodeKind::Add) {
        return std::make_shared<Add>(std::move(children));
    }
    return std::make_shared<Mul>(std::move(children));
}

void collect_add_children(const NodePtr& node, Children& out) {
    if (node->kind() == NodeKind::Add) {
        for (const auto& child : node->children()) {
            collect_add_children(child, out);
        }
    } else {
        out.push_back(node);
    }
}

void collect_mul_factors(const NodePtr& node, Children& out) {
    if (node->kind() == NodeKind::Mul) {
        for (const auto& child : node->children()) {
            collect_mul_factors(child, out);
        }
    } else {
        out.push_back(node);
    }
}

std::vector<LabelGroup> internal_label_groups(
    const std::vector<std::string>& source_indices,
    const std::set<std::string>& reserved_external,
    const std::set<std::string>& used_indices,
    const CanonicalizationMetadata& metadata, const SpaceMap& spaces) {
    std::map<std::string, std::vector<std::string>> grouped_sources;
    for (const auto& index : unique_indices(source_indices)) {
        grouped_sources[index_space(index, spaces)].push_back(index);
    }

    std::vector<LabelGroup> groups;
    for (auto& [space, sources] : grouped_sources) {
        LabelGroup group{space, sources, {}};
        const auto available = space_labels(space, metadata);
        for (const auto& candidate : available) {
            if (reserved_external.find(candidate) == reserved_external.end()) {
                group.targets.push_back(candidate);
            }
            if (group.targets.size() == sources.size()) {
                break;
            }
        }

        for (std::size_t position = group.targets.size();
             position < sources.size(); ++position) {
            std::string generated;
            do {
                generated = "__internal_" +
                            (space.empty() ? std::string("unknown") : space) + "_" +
                            std::to_string(position) + "__";
                if (used_indices.find(generated) != used_indices.end() ||
                    reserved_external.find(generated) != reserved_external.end()) {
                    generated.push_back('_');
                }
            } while (used_indices.find(generated) != used_indices.end() ||
                     reserved_external.find(generated) != reserved_external.end());
            group.targets.push_back(std::move(generated));
        }
        std::sort(group.targets.begin(), group.targets.end());
        groups.push_back(std::move(group));
    }
    return groups;
}

void enumerate_group_mappings(const std::vector<LabelGroup>& groups,
                              std::size_t group_position, IndexMap& current,
                              std::vector<IndexMap>& results) {
    if (group_position == groups.size()) {
        results.push_back(current);
        return;
    }
    const auto& group = groups[group_position];
    auto targets = group.targets;
    std::sort(targets.begin(), targets.end());
    do {
        for (std::size_t i = 0; i < group.sources.size(); ++i) {
            current[group.sources[i]] = targets[i];
        }
        enumerate_group_mappings(groups, group_position + 1, current, results);
        for (const auto& source : group.sources) {
            current.erase(source);
        }
    } while (std::next_permutation(targets.begin(), targets.end()));
}

std::vector<IndexMap> index_mappings(
    const std::vector<std::string>& source_indices,
    const std::set<std::string>& reserved_external,
    const std::set<std::string>& used_indices,
    const CanonicalizationMetadata& metadata, const SpaceMap& spaces) {
    const auto groups = internal_label_groups(source_indices, reserved_external,
                                              used_indices, metadata, spaces);
    std::vector<IndexMap> result;
    IndexMap current;
    enumerate_group_mappings(groups, 0, current, result);
    if (result.empty()) {
        result.emplace_back();
    }
    return result;
}

std::vector<Symmetry> symmetry_group(const Tensor& tensor,
                                     const CanonicalizationMetadata& metadata,
                                     const SpaceMap& spaces) {
    const auto rank = static_cast<std::size_t>(tensor.rank());
    std::vector<std::size_t> identity(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        identity[i] = i;
    }

    std::vector<std::string> tensor_spaces;
    tensor_spaces.reserve(tensor.indices().size());
    for (const auto& index : tensor.indices()) {
        const auto space = index_space(index, spaces);
        if (space.empty()) {
            tensor_spaces.clear();
            break;
        }
        tensor_spaces.push_back(space);
    }
    auto generators = metadata.symmetries.end();
    if (tensor_spaces.size() == tensor.indices().size()) {
        generators = metadata.symmetries.find({tensor.symbol(), tensor_spaces});
    }
    if (generators == metadata.symmetries.end()) {
        return {{std::move(identity), 1}};
    }

    std::map<std::vector<std::size_t>, int> group;
    group.emplace(identity, 1);
    std::vector<std::vector<std::size_t>> pending{identity};
    while (!pending.empty()) {
        auto current = std::move(pending.back());
        pending.pop_back();
        const auto current_phase = group.at(current);
        for (const auto& generator : generators->second) {
            if (generator.permutation().size() != rank) {
                throw std::invalid_argument("tensor symmetry rank mismatch");
            }
            std::vector<bool> seen(rank, false);
            for (const auto axis : generator.permutation()) {
                if (axis < 0 || static_cast<std::size_t>(axis) >= rank ||
                    seen[static_cast<std::size_t>(axis)]) {
                    throw std::invalid_argument("invalid tensor symmetry permutation");
                }
                seen[static_cast<std::size_t>(axis)] = true;
            }
            const int generator_phase =
                generator.phase() == PhaseFactor::Positive ? 1 : -1;
            std::vector<std::size_t> composed(rank);
            for (std::size_t i = 0; i < rank; ++i) {
                composed[i] = current[static_cast<std::size_t>(generator.permutation()[i])];
            }
            const int composed_phase = current_phase * generator_phase;
            const auto existing = group.find(composed);
            if (existing != group.end()) {
                if (existing->second != composed_phase) {
                    throw std::invalid_argument("inconsistent tensor symmetry phases");
                }
            } else {
                group.emplace(composed, composed_phase);
                pending.push_back(std::move(composed));
            }
        }
    }

    std::vector<Symmetry> result;
    result.reserve(group.size());
    for (auto& entry : group) {
        result.emplace_back(entry.first, entry.second);
    }
    return result;
}

NodePtr replace_tensor_indices(const Tensor& tensor, const IndexMap& mapping) {
    auto indices = tensor.indices();
    bool changed = false;
    for (auto& index : indices) {
        if (const auto found = mapping.find(index); found != mapping.end()) {
            changed = changed || found->second != index;
            index = found->second;
        }
    }
    return changed ? std::make_shared<Tensor>(tensor.symbol(), std::move(indices))
                   : std::make_shared<Tensor>(tensor.symbol(), tensor.indices());
}

Form canonicalize_node(const NodePtr& node,
                       const CanonicalizationMetadata& metadata,
                       const SpaceMap& spaces,
                       const std::set<std::string>& reserved_external,
                       const std::set<std::string>& all_used_indices);

Form canonicalize_tensor(const Tensor& tensor,
                         const CanonicalizationMetadata& metadata,
                         const SpaceMap& spaces,
                         const std::set<std::string>& reserved_external,
                         const std::set<std::string>& all_used_indices) {
    auto internal = unique_indices(tensor.internal_indices());
    const auto mappings = index_mappings(internal, reserved_external,
                                         all_used_indices, metadata, spaces);
    std::optional<Form> best;
    bool conflicting_phase = false;
    for (const auto& mapping : mappings) {
        auto renamed = replace_tensor_indices(tensor, mapping);
        const auto* renamed_tensor = dynamic_cast<const Tensor*>(renamed.get());
        for (const auto& [permutation, phase] : symmetry_group(*renamed_tensor, metadata, spaces)) {
            std::vector<std::string> axes;
            axes.reserve(permutation.size());
            for (const auto position : permutation) {
                axes.push_back(renamed_tensor->indices()[position]);
            }
            auto candidate = std::make_shared<Tensor>(renamed_tensor->symbol(), std::move(axes));
            if (!best || node_less(candidate, best->node)) {
                best = Form{std::move(candidate), phase};
                conflicting_phase = false;
            } else if (!node_less(best->node, candidate)) {
                if (best->phase != phase) {
                    conflicting_phase = true;
                }
            }
        }
    }
    if (conflicting_phase) {
        return {std::make_shared<Scalar>(0.0), 1};
    }
    return *best;
}

NodePtr normalize_product(Children factors, double coefficient) {
    if (coefficient != 1.0 || factors.empty()) {
        factors.push_back(std::make_shared<Scalar>(coefficient));
    }
    NodePtr result = flatten(std::make_shared<Mul>(std::move(factors)));
    const auto is_product = [](const NodePtr& node) {
        return node->kind() == NodeKind::Mul;
    };
    result = albert::apply(
        result, [](const NodePtr& node) { return fold_scalars(node); }, is_product);
    result = albert::apply(
        result, [](const NodePtr& node) { return prune_scalars(node); }, is_product);
    result = albert::apply(
        result, [](const NodePtr& node) { return squeeze(node); }, is_product);

    const auto* product = dynamic_cast<const Mul*>(result.get());
    if (!product) {
        return result;
    }
    Children non_scalars;
    NodePtr scalar;
    for (const auto& factor : product->children()) {
        if (factor->kind() == NodeKind::Scalar) {
            scalar = factor;
        } else {
            non_scalars.push_back(factor);
        }
    }
    std::sort(non_scalars.begin(), non_scalars.end(), node_less);
    if (scalar) {
        non_scalars.push_back(std::move(scalar));
    }
    if (non_scalars.size() == 1) {
        return non_scalars.front();
    }
    return std::make_shared<Mul>(std::move(non_scalars));
}

NodePtr apply_phase(const NodePtr& node, int phase) {
    if (phase == 1) {
        return node;
    }
    if (const auto* scalar = dynamic_cast<const Scalar*>(node.get())) {
        return std::make_shared<Scalar>(-scalar->value());
    }
    return normalize_product(Children{node}, -1.0);
}

Form canonicalize_add(const NodePtr& node,
                      const CanonicalizationMetadata& metadata,
                      const SpaceMap& spaces,
                      const std::set<std::string>& reserved_external,
                      const std::set<std::string>& all_used_indices) {
    Children terms;
    collect_add_children(node, terms);
    std::vector<Form> forms(terms.size());
    std::vector<std::exception_ptr> errors(terms.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (terms.size() >= 32)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(terms.size()); ++i) {
        try {
            forms[static_cast<std::size_t>(i)] = canonicalize_node(
                terms[static_cast<std::size_t>(i)], metadata, spaces,
                reserved_external, all_used_indices);
        } catch (...) {
            errors[static_cast<std::size_t>(i)] = std::current_exception();
        }
    }

    Children canonical_terms;
    for (std::size_t i = 0; i < forms.size(); ++i) {
        if (errors[i]) {
            std::rethrow_exception(errors[i]);
        }
        auto signed_term = apply_phase(forms[i].node, forms[i].phase);
        collect_add_children(signed_term, canonical_terms);
    }
    std::sort(canonical_terms.begin(), canonical_terms.end(), node_less);
    if (canonical_terms.empty()) {
        return {std::make_shared<Add>(Children{}), 1};
    }
    if (canonical_terms.size() == 1) {
        return {canonical_terms.front(), 1};
    }
    return {std::make_shared<Add>(std::move(canonical_terms)), 1};
}

Form canonicalize_mul(const NodePtr& node,
                      const CanonicalizationMetadata& metadata,
                      const SpaceMap& spaces,
                      const std::set<std::string>& reserved_external,
                      const std::set<std::string>& all_used_indices) {
    Children factors;
    collect_mul_factors(node, factors);
    std::unordered_map<std::string, std::size_t> counts;
    std::vector<std::string> internal_order;
    for (const auto& factor : factors) {
        for (const auto& index : factor->external_indices()) {
            if (counts.find(index) == counts.end()) {
                internal_order.push_back(index);
            }
            ++counts[index];
        }
    }
    for (const auto& [index, count] : counts) {
        if (count > 2) {
            throw std::invalid_argument("Einstein index occurs more than twice");
        }
    }
    std::vector<std::string> internal;
    for (const auto& index : internal_order) {
        if (counts[index] == 2) {
            internal.push_back(index);
        }
    }
    const auto mappings = index_mappings(internal, reserved_external,
                                         all_used_indices, metadata, spaces);

    std::optional<NodePtr> best;
    for (const auto& mapping : mappings) {
        Children canonical_factors;
        canonical_factors.reserve(factors.size());
        double coefficient = 1.0;
        bool zero = false;
        for (const auto& factor : factors) {
            auto renamed = map_external_indices(factor, mapping);
            auto form = canonicalize_node(renamed, metadata, spaces,
                                          reserved_external, all_used_indices);
            if (form.phase == -1) {
                coefficient = -coefficient;
            }
            if (const auto* scalar = dynamic_cast<const Scalar*>(form.node.get())) {
                coefficient *= scalar->value();
                zero = zero || scalar->value() == 0.0;
            } else {
                canonical_factors.push_back(std::move(form.node));
            }
        }
        auto candidate = zero ? std::make_shared<Scalar>(0.0)
                              : normalize_product(std::move(canonical_factors), coefficient);
        if (!best || node_less(candidate, *best)) {
            best = std::move(candidate);
        }
    }
    return {*best, 1};
}

Form canonicalize_node(const NodePtr& node,
                       const CanonicalizationMetadata& metadata,
                       const SpaceMap& spaces,
                       const std::set<std::string>& reserved_external,
                       const std::set<std::string>& all_used_indices) {
    if (const auto* tensor = dynamic_cast<const Tensor*>(node.get())) {
        return canonicalize_tensor(*tensor, metadata, spaces,
                                   reserved_external, all_used_indices);
    }
    if (node->kind() == NodeKind::Scalar) {
        return {node, 1};
    }
    if (node->kind() == NodeKind::Add) {
        return canonicalize_add(node, metadata, spaces,
                                reserved_external, all_used_indices);
    }
    return canonicalize_mul(node, metadata, spaces,
                            reserved_external, all_used_indices);
}

}  // namespace

CanonicalizedNode canonicalize_with_mapping(
    const NodePtr& node, const CanonicalizationMetadata& metadata) {
    if (!node) {
        throw std::invalid_argument("AST node cannot be null");
    }
    const auto spaces = make_space_map(metadata);
    std::set<std::string> all_used_indices;
    collect_indices(node, all_used_indices);
    const auto external = external_mappings(node, spaces, all_used_indices);

    std::optional<CanonicalizedNode> best;
    for (const auto& mapping : external.mappings) {
        auto mapped = albert::map_indices(node, mapping);
        auto mapped_spaces = spaces;
        for (const auto& [placeholder, space] : external.spaces) {
            mapped_spaces.emplace(placeholder, space);
        }
        const auto root_external = mapped->external_indices();
        std::set<std::string> reserved_external(root_external.begin(),
                                                 root_external.end());
        std::map<std::string, std::size_t> external_counts;
        for (const auto& index : root_external) {
            const auto space = external.spaces.find(index);
            if (space != external.spaces.end()) {
                ++external_counts[space->second];
            }
        }
        for (const auto& [space, count] : external_counts) {
            const auto labels = space_labels(space, metadata);
            for (std::size_t i = 0; i < std::min(count, labels.size()); ++i) {
                reserved_external.insert(labels[i]);
            }
        }
        auto form = canonicalize_node(mapped, metadata, mapped_spaces,
                                      reserved_external, all_used_indices);
        auto phased = apply_phase(form.node, form.phase);
        for (const auto& final_mapping : final_external_mappings(
                 phased, external.spaces, metadata)) {
            auto candidate = albert::map_indices(phased, final_mapping);
            if (!best || node_less(candidate, best->node)) {
                std::unordered_map<std::string, std::string> combined;
                for (const auto& [source, placeholder] : mapping) {
                    const auto target = final_mapping.find(placeholder);
                    if (target != final_mapping.end()) {
                        combined.emplace(source, target->second);
                    }
                }
                best = CanonicalizedNode{std::move(candidate), std::move(combined)};
            }
        }
    }
    return *best;
}

NodePtr canonicalize(const NodePtr& node,
                     const CanonicalizationMetadata& metadata) {
    return canonicalize_with_mapping(node, metadata).node;
}

}  // namespace albert
