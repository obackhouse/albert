#include "algebra.h"
#include "canonicalize.h"
#include "cost.h"
#include "expression.h"
#include "permutation.h"
#include "search.h"
#include "simplify.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
        .def_readonly("multiplications", &albert::CostEstimate::multiplications)
        .def_readonly("additions", &albert::CostEstimate::additions)
        .def_readonly("reads", &albert::CostEstimate::reads)
        .def_readonly("writes", &albert::CostEstimate::writes)
        .def_readonly("total", &albert::CostEstimate::total);
    module.def("canonical_key", &albert::canonical_key, py::arg("node"));
    module.def(
        "estimate_cost",
        [](const albert::NodePtr& node, const py::object& metadata,
           const py::object& cost_model) {
            const auto spaces = metadata.attr("spaces")
                                    .cast<albert::IndexSpaces>();
            const auto sizes = metadata.attr("sizes")
                                   .cast<albert::SpaceSizes>();
            albert::CostModel model;
            if (!cost_model.is_none()) {
                model.multiply = cost_model.attr("multiply").cast<double>();
                model.add = cost_model.attr("add").cast<double>();
                model.read = cost_model.attr("read").cast<double>();
                model.write = cost_model.attr("write").cast<double>();
            }
            return albert::estimate_cost(node, spaces, sizes, model);
        },
        py::arg("node"), py::arg("metadata"), py::arg("cost_model") = py::none(),
        "Estimate arithmetic operations and memory traffic for an AST.");
    module.def(
        "canonicalize",
        [](const albert::NodePtr& node, const py::object& metadata) {
            albert::CanonicalizationMetadata canonical_metadata;
            if (!metadata.is_none()) {
                canonical_metadata.spaces = metadata.attr("spaces")
                                                .cast<std::unordered_map<
                                                    std::string,
                                                    std::vector<std::string>>>();
                canonical_metadata.symmetries = metadata.attr("symmetries")
                                                    .cast<std::unordered_map<
                                                        std::string,
                                                        std::vector<albert::Permutation>>>();
            }
            return albert::canonicalize(node, canonical_metadata);
        },
        py::arg("node"), py::arg("metadata") = py::none());
    module.def("expand", &albert::expand, py::arg("node"));
    module.def("collect", &albert::collect, py::arg("node"));
    module.def("squeeze", &albert::squeeze, py::arg("node"));
    module.def("remove_neutral_elements", &albert::remove_neutral_elements,
               py::arg("node"));
    module.def("simplify", &albert::simplify, py::arg("node"));

    py::class_<albert::Node, albert::NodePtr>(module, "Node")
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
        .def_property_readonly("lhs", &albert::Expression::lhs)
        .def_property_readonly("rhs", &albert::Expression::rhs);

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
}
