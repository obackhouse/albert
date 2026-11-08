#include "algebra.h"
#include "expression.h"
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
    module.def("canonical_key", &albert::canonical_key, py::arg("node"));
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
