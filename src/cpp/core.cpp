#include "algebra.h"
#include "canonicalize.h"
#include "cost.h"
#include "cse.h"
#include "evaluate.h"
#include "substitute.h"
#include "expression.h"
#include "permutation.h"
#include "parenthesize.h"
#include "search.h"
#include "simplify.h"
#include "sort.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace py = pybind11;

namespace {

albert::SearchOrder parse_search_order(const std::string& order) {
    if (order == "preorder") {
        return albert::SearchOrder::PreOrder;
    }
    if (order == "postorder") {
        return albert::SearchOrder::PostOrder;
    }
    throw py::value_error("order must be SearchOrder.PREORDER or SearchOrder.POSTORDER");
}

Py_hash_t python_hash(std::uint64_t hash) {
    const auto result = static_cast<Py_hash_t>(hash);
    return result == -1 ? -2 : result;
}

albert::ScalingCaps parse_scaling_caps(const py::object& mapping) {
    albert::ScalingCaps result;
    for (const auto& item : mapping.cast<py::dict>()) {
        std::vector<std::string> spaces;
        if (py::isinstance<py::str>(item.first)) {
            spaces.push_back(item.first.cast<std::string>());
        } else if (py::isinstance<py::tuple>(item.first)) {
            spaces = item.first.cast<std::vector<std::string>>();
        } else {
            throw py::value_error("scaling limit keys must be a space name or tuple of names");
        }
        if (spaces.empty()) {
            throw py::value_error("scaling limit space tuples cannot be empty");
        }
        const auto maximum = item.second.cast<std::size_t>();
        result.emplace_back(std::move(spaces), maximum);
    }
    return result;
}

albert::ScalingLimits parse_scaling_limits(const py::object& limits) {
    if (limits.is_none()) {
        return {};
    }
    return {parse_scaling_caps(limits.attr("time")),
            parse_scaling_caps(limits.attr("memory"))};
}

albert::PathRequests parse_path_requests(const py::object& value) {
    albert::PathRequests result;
    result.bounds = value.attr("bounds").cast<std::vector<std::size_t>>();
    for (const auto& request : value.attr("requests").cast<py::tuple>()) {
        result.requests.push_back({
            request.attr("exhaustive").cast<std::size_t>(),
            request.attr("branch").cast<std::size_t>(),
            request.attr("greedy").cast<std::size_t>(),
            request.attr("dynamic").cast<bool>()});
    }
    if (result.requests.size() != result.bounds.size() + 1 ||
        !std::is_sorted(result.bounds.begin(), result.bounds.end()) ||
        std::adjacent_find(result.bounds.begin(), result.bounds.end()) != result.bounds.end()) {
        throw py::value_error("PathRequests requires sorted bounds and one extra request");
    }
    return result;
}

albert::CanonicalizationMetadata parse_canonicalization_metadata(
    const py::object& metadata) {
    albert::CanonicalizationMetadata result;
    if (!metadata.is_none()) {
        result.spaces = metadata.attr("indices")
                            .cast<std::unordered_map<std::string,
                                                     std::vector<std::string>>>();
        result.symmetries = metadata.attr("symmetries")
                                .cast<std::unordered_map<
                                    std::string, std::vector<albert::Permutation>>>();
    }
    return result;
}

albert::CostFactorAggregation parse_cost_factor_aggregation(const std::string& value) {
    if (value == "product") {
        return albert::CostFactorAggregation::Product;
    }
    if (value == "maximum") {
        return albert::CostFactorAggregation::Maximum;
    }
    throw py::value_error("cost_factor_aggregation must be 'product' or 'maximum'");
}

std::vector<std::string> parse_space_argument(const py::object& space) {
    if (py::isinstance<py::str>(space)) {
        return {space.cast<std::string>()};
    }
    if (py::isinstance<py::tuple>(space)) {
        auto result = space.cast<std::vector<std::string>>();
        if (!result.empty()) {
            return result;
        }
    }
    throw py::value_error("space must be a space name or nonempty tuple of names");
}

py::object resolve_encoder(py::object encoder,
                           const std::string& default_encoder = "ReadableEncoder") {
    const auto encode_module = py::module_::import("albert.encode");
    const auto encoder_base = encode_module.attr("Encoder");
    if (encoder.is_none()) {
        return encode_module.attr(default_encoder.c_str())();
    }
    if (py::isinstance<py::str>(encoder)) {
        const auto method = py::module_::import("albert.options")
                                .attr("EncodingMethod")(encoder);
        return method.attr("encoder_type")();
    }
    if (py::isinstance(encoder, encoder_base)) {
        return encoder;
    }
    if (PyType_Check(encoder.ptr()) &&
        py::module_::import("builtins").attr("issubclass")(encoder, encoder_base).cast<bool>()) {
        return encoder();
    }
    throw py::type_error("encoder must be an Encoder instance or Encoder subclass");
}

py::object resolve_decoder(py::object decoder) {
    const auto decode_module = py::module_::import("albert.decode");
    const auto decoder_base = decode_module.attr("Decoder");
    if (decoder.is_none()) {
        return decode_module.attr("JsonDecoder")();
    }
    if (py::isinstance<py::str>(decoder)) {
        const auto method = py::module_::import("albert.options")
                                .attr("DecodingMethod")(decoder);
        return method.attr("decoder_type")();
    }
    if (py::isinstance(decoder, decoder_base)) {
        return decoder;
    }
    if (PyType_Check(decoder.ptr()) &&
        py::module_::import("builtins").attr("issubclass")(decoder, decoder_base).cast<bool>()) {
        return decoder();
    }
    throw py::type_error("decoder must be a Decoder instance or Decoder subclass");
}

py::object readable_repr_encoder() {
    return py::module_::import("albert.encode")
        .attr("ReadableEncoder")(py::none(), py::make_tuple("(", ")"), ",");
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ extension module for albert.";

    const auto node_kind_enum = py::module_::import("albert.options").attr("NodeKind");
    const auto phase_factor_enum = py::module_::import("albert.options").attr("PhaseFactor");

    py::class_<albert::CanonicalKey>(module, "CanonicalKey")
        .def("__eq__",
             [](const albert::CanonicalKey& self,
                const py::object& other) -> py::object {
                 if (!py::isinstance<albert::CanonicalKey>(other)) {
                     return py::reinterpret_borrow<py::object>(Py_NotImplemented);
                 }
                 return py::bool_(self == other.cast<albert::CanonicalKey>());
             },
             py::is_operator())
        .def("__hash__", [](const albert::CanonicalKey& key) {
            return python_hash(key.hash());
        });
    py::class_<albert::CostEstimate>(module, "CostEstimate",
                                     "Operation and traffic counts with their weighted total.")
        .def(py::init([](double multiplications, double additions, double reads,
                         double writes, double total) {
                 return albert::CostEstimate{multiplications, additions, reads,
                                             writes, total};
             }),
             py::arg("multiplications") = 0.0, py::arg("additions") = 0.0,
             py::arg("reads") = 0.0, py::arg("writes") = 0.0,
             py::arg("total") = 0.0)
        .def_readonly("multiplications", &albert::CostEstimate::multiplications)
        .def_readonly("additions", &albert::CostEstimate::additions)
        .def_readonly("reads", &albert::CostEstimate::reads)
        .def_readonly("writes", &albert::CostEstimate::writes)
        .def_readonly("total", &albert::CostEstimate::total);
    py::class_<albert::ParenthesizationCandidate>(module, "ParenthesizationCandidate")
        .def_readonly("path", &albert::ParenthesizationCandidate::path)
        .def_readonly("cost", &albert::ParenthesizationCandidate::cost);
    module.def("canonical_key", &albert::canonical_key, py::arg("node"));
    module.def(
        "estimate_cost",
        [](const albert::NodePtr& node, const py::object& metadata,
           const py::object& cost_model) {
            const auto spaces = metadata.attr("indices")
                                    .cast<albert::IndexSpaces>();
            const auto sizes = metadata.attr("sizes")
                                   .cast<albert::SpaceSizes>();
            albert::CostModel model;
            albert::CostFactors model_cost_factors;
            if (!cost_model.is_none()) {
                model.multiply = cost_model.attr("multiply").cast<double>();
                model.add = cost_model.attr("add").cast<double>();
                model.read = cost_model.attr("read").cast<double>();
                model.write = cost_model.attr("write").cast<double>();
                model.apply_cost_factors =
                    cost_model.attr("apply_cost_factors").cast<bool>();
                model.cost_factor_aggregation = parse_cost_factor_aggregation(
                    cost_model.attr("cost_factor_aggregation").cast<std::string>());
                model_cost_factors = cost_model.attr("cost_factors").cast<albert::CostFactors>();
            }
            return albert::estimate_cost(node, spaces, sizes, model, model_cost_factors);
        },
        py::arg("node"), py::arg("metadata"), py::arg("cost_model") = py::none(),
        "Estimate arithmetic operations and memory traffic for an AST.");
    module.def(
        "canonicalize",
        [](const albert::NodePtr& node, const py::object& metadata) {
            return albert::canonicalize(
                node, parse_canonicalization_metadata(metadata));
        },
        py::arg("node"), py::arg("metadata") = py::none());
    module.def(
        "canonicalize",
        [](const albert::Expression& expression, const py::object& metadata) {
            auto canonicalized = albert::canonicalize_with_mapping(
                expression.rhs(), parse_canonicalization_metadata(metadata));
            auto lhs_indices = expression.lhs()->indices();
            for (auto& index : lhs_indices) {
                if (const auto found = canonicalized.external_index_mapping.find(index);
                    found != canonicalized.external_index_mapping.end()) {
                    index = found->second;
                }
            }
            auto lhs = std::make_shared<albert::Tensor>(
                expression.lhs()->symbol(), std::move(lhs_indices));
            return albert::Expression(std::move(lhs), std::move(canonicalized.node));
        },
        py::arg("node"), py::arg("metadata") = py::none());
    module.def("expand", &albert::expand, py::arg("node"));
    module.def("flatten", &albert::flatten, py::arg("node"));
    module.def("collect", &albert::collect, py::arg("node"));
    module.def("fold_scalars", &albert::fold_scalars, py::arg("node"));
    module.def("squeeze", &albert::squeeze, py::arg("node"));
    module.def("prune_scalars", &albert::prune_scalars, py::arg("node"));
    module.def("simplify", &albert::simplify, py::arg("node"));
    module.def(
        "parenthesization_candidates",
        [](const albert::Mul& node, const py::object& metadata,
           const py::object& scaling_limits, std::size_t max_candidates,
           std::size_t exhaustive_limit, const py::object& cost_model,
           const py::object& path_requests) {
            const auto spaces = metadata.attr("indices").cast<albert::IndexSpaces>();
            const auto sizes = metadata.attr("sizes").cast<albert::SpaceSizes>();
            albert::CostModel model;
            albert::CostFactors model_cost_factors;
            if (!cost_model.is_none()) {
                model.multiply = cost_model.attr("multiply").cast<double>();
                model.add = cost_model.attr("add").cast<double>();
                model.read = cost_model.attr("read").cast<double>();
                model.write = cost_model.attr("write").cast<double>();
                model.apply_cost_factors =
                    cost_model.attr("apply_cost_factors").cast<bool>();
                model.cost_factor_aggregation = parse_cost_factor_aggregation(
                    cost_model.attr("cost_factor_aggregation").cast<std::string>());
                model_cost_factors = cost_model.attr("cost_factors").cast<albert::CostFactors>();
            }
            const auto limits = parse_scaling_limits(scaling_limits);
            if (!path_requests.is_none()) {
                const auto requests = parse_path_requests(path_requests);
                return albert::parenthesization_candidates(
                    node, spaces, sizes, limits,
                    requests.for_size(node.children().size()), max_candidates, model,
                    model_cost_factors);
            }
            return albert::parenthesization_candidates(
                node, spaces, sizes, limits, max_candidates, exhaustive_limit, model,
                model_cost_factors);
        },
        py::arg("node"), py::arg("metadata"),
        py::arg("scaling_limits") = py::none(), py::arg("max_candidates") = 8,
        py::arg("exhaustive_limit") = 5, py::arg("cost_model") = py::none(),
        py::arg("path_requests") = py::none());
    module.def("apply_parenthesization", &albert::apply_parenthesization,
               py::arg("node"), py::arg("candidate"));

    py::class_<albert::Node, albert::NodePtr>(module, "Node")
        .def("__repr__", [](const albert::NodePtr& self) {
            const auto encoder = readable_repr_encoder();
            return encoder.attr("format_node")(py::cast(self)).cast<std::string>();
        })
        .def("format", [](const albert::NodePtr& self, py::object encoder) {
            const auto resolved = resolve_encoder(std::move(encoder));
            return resolved.attr("format_node")(py::cast(self)).cast<std::string>();
        }, py::arg("encoder") = py::none())
        .def("encode", [](const albert::NodePtr& self, py::object encoder) {
            const auto resolved = resolve_encoder(std::move(encoder), "JsonEncoder");
            resolved.attr("encode_node")(py::cast(self));
        }, py::arg("encoder") = py::none())
        .def("__iter__", [](const albert::NodePtr& self) {
            return py::iter(py::cast(self->children()));
        })
        .def("__contains__", [](const albert::NodePtr& self, const py::object& candidate) {
            if (!py::isinstance<albert::Node>(candidate)) {
                return false;
            }
            const auto child = candidate.cast<albert::NodePtr>();
            return std::any_of(self->children().begin(), self->children().end(),
                               [&child](const albert::NodePtr& item) {
                                   return item->structurally_equal(*child);
                               });
        })
        .def_property_readonly("kind", [node_kind_enum](const albert::Node& node) -> py::object {
            switch (node.kind()) {
                case albert::NodeKind::Scalar:
                    return node_kind_enum.attr("SCALAR");
                case albert::NodeKind::Tensor:
                    return node_kind_enum.attr("TENSOR");
                case albert::NodeKind::Add:
                    return node_kind_enum.attr("ADD");
                case albert::NodeKind::Mul:
                    return node_kind_enum.attr("MUL");
            }
            throw py::value_error("unknown node kind");
        })
        .def("__eq__",
             [](const albert::NodePtr& self, const py::object& other) -> py::object {
                 if (!py::isinstance<albert::Node>(other)) {
                     return py::reinterpret_borrow<py::object>(Py_NotImplemented);
                 }
                 const auto other_node = other.cast<albert::NodePtr>();
                 return py::bool_(self->structurally_equal(*other_node));
             },
             py::is_operator())
        .def("__hash__", [](const albert::NodePtr& self) {
            return python_hash(self->structural_hash());
        })
        .def_property_readonly("external_indices", &albert::Node::external_indices)
        .def_property_readonly("internal_indices", &albert::Node::internal_indices)
        .def("map_indices",
             [](const albert::NodePtr& self,
                const std::unordered_map<std::string, std::string>& mapping) {
                 return albert::map_indices(self, mapping);
             },
             py::arg("mapping"))
        .def("permute_indices",
             [](const albert::NodePtr& self, const py::object& permutation) {
                 if (py::isinstance<albert::Permutation>(permutation)) {
                     return albert::permute_indices(
                         self, permutation.cast<albert::Permutation>());
                 }
                 return albert::permute_indices(
                     self, permutation.cast<std::vector<int>>());
             },
             py::arg("permutation"))
        .def_property_readonly("children", [](const albert::Node& node) {
            return node.children();
        })
        .def("search",
             [](const albert::NodePtr& self, py::function predicate,
                std::optional<std::size_t> max_depth, const std::string& order) {
                 auto callback = [&predicate](const albert::NodePtr& node) {
                     return predicate(node).cast<bool>();
                 };
                 return albert::search(self, callback, max_depth, parse_search_order(order));
             },
             py::arg("predicate"), py::arg("max_depth") = std::nullopt,
             py::arg("order") = "preorder")
        .def("find",
             [](const albert::NodePtr& self, py::function predicate,
                std::optional<std::size_t> max_depth, const std::string& order) {
                 auto callback = [&predicate](const albert::NodePtr& node) {
                     return predicate(node).cast<bool>();
                 };
                 return albert::find(self, callback, max_depth, parse_search_order(order));
             },
             py::arg("predicate"), py::arg("max_depth") = std::nullopt,
             py::arg("order") = "preorder")
        .def("delete",
             [](const albert::NodePtr& self, py::function predicate,
                std::optional<std::size_t> max_depth) {
                 auto callback = [&predicate](const albert::NodePtr& node) {
                     return predicate(node).cast<bool>();
                 };
                 return albert::delete_nodes(self, callback, max_depth);
             },
             py::arg("predicate"), py::arg("max_depth") = std::nullopt)
        .def("apply",
             [](const albert::NodePtr& self, py::function filter,
                py::object predicate, std::optional<std::size_t> max_depth) {
                 auto transform = [&filter](const albert::NodePtr& node) {
                     return filter(node).cast<albert::NodePtr>();
                 };
                 auto callback = [&predicate](const albert::NodePtr& node) {
                     return predicate.is_none() || predicate(node).cast<bool>();
                 };
                 return albert::apply(self, transform, callback, max_depth);
             },
             py::arg("filter"), py::arg("predicate") = py::none(),
             py::arg("max_depth") = std::nullopt)
        .def("__add__",
             [](albert::NodePtr lhs, albert::NodePtr rhs) {
                 return std::move(lhs) + std::move(rhs);
             },
             py::is_operator())
        .def("__add__",
             [](albert::NodePtr lhs, double rhs) {
                 return std::move(lhs) + rhs;
             },
             py::is_operator())
        .def("__radd__",
             [](albert::NodePtr rhs, double lhs) {
                 return lhs + std::move(rhs);
             },
             py::is_operator())
        .def("__sub__",
             [](albert::NodePtr lhs, albert::NodePtr rhs) {
                 return std::move(lhs) - std::move(rhs);
             },
             py::is_operator())
        .def("__sub__",
             [](albert::NodePtr lhs, double rhs) {
                 return std::move(lhs) - rhs;
             },
             py::is_operator())
        .def("__rsub__",
             [](albert::NodePtr rhs, double lhs) {
                 return lhs - std::move(rhs);
             },
             py::is_operator())
        .def("__mul__",
             [](albert::NodePtr lhs, albert::NodePtr rhs) {
                 return std::move(lhs) * std::move(rhs);
             },
             py::is_operator())
        .def("__mul__",
             [](albert::NodePtr lhs, double rhs) {
                 return std::move(lhs) * rhs;
             },
             py::is_operator())
        .def("__rmul__",
             [](albert::NodePtr rhs, double lhs) {
                 return lhs * std::move(rhs);
             },
             py::is_operator())
        .def("__pos__",
             [](albert::NodePtr value) { return +std::move(value); },
             py::is_operator())
        .def("__neg__",
             [](albert::NodePtr value) { return -std::move(value); },
             py::is_operator());

    const auto decode_method = py::cpp_function(
        [](const py::object& cls, const std::string& value, py::object decoder) {
            const auto resolved = resolve_decoder(std::move(decoder));
            const auto result = resolved.attr("decode_node")(value);
            if (!py::isinstance(result, cls)) {
                throw py::type_error("decoder returned a node incompatible with the requested class");
            }
            return result;
        },
        py::arg("cls"), py::arg("value"), py::arg("decoder") = py::none());
    module.attr("Node").attr("decode") =
        py::module_::import("builtins").attr("classmethod")(decode_method);

    py::class_<albert::Scalar, albert::Node, std::shared_ptr<albert::Scalar>>(
        module, "Scalar")
        .def(py::init<double>())
        .def_property_readonly("value", &albert::Scalar::value);

    py::class_<albert::Tensor, albert::Node, std::shared_ptr<albert::Tensor>>(
        module, "Tensor")
        .def(py::init<std::string, std::vector<std::string>>())
        .def_property_readonly("symbol", &albert::Tensor::symbol)
        .def_property_readonly("rank", &albert::Tensor::rank)
        .def_property_readonly("indices", &albert::Tensor::indices);

    py::class_<albert::Expression>(module, "Expression")
        .def(py::init<std::shared_ptr<albert::Tensor>, albert::NodePtr>(),
             py::arg("lhs"), py::arg("rhs"))
        .def("__repr__", [](const albert::Expression& self) {
            const auto encoder = readable_repr_encoder();
            return encoder.attr("format_expression")(py::cast(self)).cast<std::string>();
        })
        .def("format", [](const albert::Expression& self, py::object encoder) {
            const auto resolved = resolve_encoder(std::move(encoder));
            return resolved.attr("format_expression")(py::cast(self)).cast<std::string>();
        }, py::arg("encoder") = py::none())
        .def("encode", [](const albert::Expression& self, py::object encoder) {
            const auto resolved = resolve_encoder(std::move(encoder));
            resolved.attr("encode_expression")(py::cast(self));
        }, py::arg("encoder") = py::none())
        .def_property_readonly("lhs", &albert::Expression::lhs)
        .def_property_readonly("rhs", &albert::Expression::rhs);

    module.def(
        "scaling_exponent",
        [](const py::object& value, const py::object& space,
           const std::string& which, const py::object& metadata) {
            albert::NodePtr node;
            if (py::isinstance<albert::Expression>(value)) {
                node = value.cast<const albert::Expression&>().rhs();
            } else if (py::isinstance<albert::Node>(value)) {
                node = value.cast<albert::NodePtr>();
            } else {
                throw py::type_error("node must be a Node or Expression");
            }
            const auto spaces = metadata.attr("indices")
                                    .cast<albert::IndexSpaces>();
            return albert::scaling_exponent(
                node, spaces, parse_space_argument(space), which);
        },
        py::arg("node"), py::arg("space"), py::arg("which"),
        py::arg("metadata"),
        "Return the maximum index-space exponent for a node or expression RHS.");

    const auto expression_decode_method = py::cpp_function(
        [](const py::object& cls, const std::string& value, py::object decoder) {
            const auto resolved = resolve_decoder(std::move(decoder));
            const auto result = resolved.attr("decode_expression")(value);
            if (!py::isinstance(result, cls)) {
                throw py::type_error(
                    "decoder returned an expression incompatible with the requested class");
            }
            return result;
        },
        py::arg("cls"), py::arg("value"), py::arg("decoder") = py::none());
    module.attr("Expression").attr("decode") =
        py::module_::import("builtins").attr("classmethod")(expression_decode_method);

    py::class_<albert::Permutation>(module, "Permutation")
        .def(py::init([](std::vector<int> permutation, int phase) {
                 if (phase != 1 && phase != -1) {
                     throw py::value_error("phase must be PhaseFactor.POSITIVE or NEGATIVE");
                 }
                 return albert::Permutation(
                     std::move(permutation),
                     static_cast<albert::PhaseFactor>(phase));
             }),
             py::arg("permutation"), py::arg("phase") = 1)
        .def_property_readonly("permutation", &albert::Permutation::permutation)
        .def_property_readonly("phase", [phase_factor_enum](const albert::Permutation& permutation) {
            return phase_factor_enum.attr(permutation.phase() == albert::PhaseFactor::Positive
                                              ? "POSITIVE"
                                              : "NEGATIVE");
        });

    py::class_<albert::Add, albert::Node, std::shared_ptr<albert::Add>>(
        module, "Add")
        .def(py::init<albert::Children>())
        .def_property_readonly("arity", &albert::Add::arity)
        .def_property_readonly("children", &albert::Add::children);

    py::class_<albert::Mul, albert::Node, std::shared_ptr<albert::Mul>>(
        module, "Mul")
        .def(py::init<albert::Children>())
        .def_property_readonly("arity", &albert::Mul::arity)
        .def_property_readonly("children", &albert::Mul::children);

    module.def(
        "topological_sort",
        [](std::vector<albert::Expression> expressions,
           const py::object& metadata) {
            if (metadata.is_none()) {
                return albert::topological_sort(std::move(expressions));
            }
            const auto spaces = metadata.attr("indices")
                                    .cast<albert::IndexSpaces>();
            return albert::topological_sort(std::move(expressions), spaces);
        },
        py::arg("expressions"), py::arg("metadata") = py::none(),
        "Return expressions in stable dependency order.");
    module.def("substitute_expressions",
               [](std::vector<albert::Expression> expressions,
                  const py::object& metadata) {
                   const auto spaces = metadata.attr("indices")
                                           .cast<albert::IndexSpaces>();
                   return albert::substitute_expressions(std::move(expressions), spaces);
               },
               py::arg("expressions"), py::arg("metadata"),
               "Inline expression definitions into their dependents.");
    module.def(
        "common_subexpression_elimination",
        [](std::vector<albert::Expression> expressions, const py::object& metadata,
           const py::object& scaling_limits, const py::object& cost_model,
           const std::string& intermediate_format, std::size_t max_path_samples,
           std::size_t max_bicliques, std::size_t exhaustive_limit,
           const py::object& path_requests) {
            albert::CSEOptions options;
            options.intermediate_format = intermediate_format;
            options.max_path_samples = max_path_samples;
            options.max_bicliques = max_bicliques;
            options.exhaustive_limit = exhaustive_limit;
            if (!path_requests.is_none()) {
                options.custom_path_requests = true;
                options.path_requests = parse_path_requests(path_requests);
            }
            options.scaling_limits = parse_scaling_limits(scaling_limits);
            albert::CostFactors model_cost_factors;
            if (!cost_model.is_none()) {
                options.cost_model.multiply = cost_model.attr("multiply").cast<double>();
                options.cost_model.add = cost_model.attr("add").cast<double>();
                options.cost_model.read = cost_model.attr("read").cast<double>();
                options.cost_model.write = cost_model.attr("write").cast<double>();
                options.cost_model.apply_cost_factors =
                    cost_model.attr("apply_cost_factors").cast<bool>();
                options.cost_model.cost_factor_aggregation = parse_cost_factor_aggregation(
                    cost_model.attr("cost_factor_aggregation").cast<std::string>());
                model_cost_factors = cost_model.attr("cost_factors").cast<albert::CostFactors>();
            }
            albert::CanonicalizationMetadata canonicalization;
            canonicalization.spaces = metadata.attr("indices")
                                          .cast<albert::IndexSpaces>();
            canonicalization.symmetries = metadata.attr("symmetries")
                                              .cast<std::unordered_map<
                                                  std::string,
                                                  std::vector<albert::Permutation>>>();
            return albert::common_subexpression_elimination(
                std::move(expressions), canonicalization.spaces,
                metadata.attr("sizes").cast<albert::SpaceSizes>(), model_cost_factors,
                canonicalization, options);
        },
        py::arg("expressions"), py::arg("metadata"),
        py::arg("scaling_limits") = py::none(), py::arg("cost_model") = py::none(),
        py::arg("intermediate_format") = "__im{}__",
        py::arg("max_path_samples") = 19, py::arg("max_bicliques") = 8,
        py::arg("exhaustive_limit") = 4,
        py::arg("path_requests") = py::none(),
        "Factor profitable common subexpressions across one or more expressions.");

    albert::bind_evaluate_expression(module);

}
