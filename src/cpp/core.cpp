#include "algebra.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ extension module for albert.";

    py::enum_<albert::NodeKind>(module, "NodeKind")
        .value("SCALAR", albert::NodeKind::Scalar)
        .value("TENSOR", albert::NodeKind::Tensor)
        .value("ADD", albert::NodeKind::Add)
        .value("MUL", albert::NodeKind::Mul);

    py::class_<albert::Node, albert::NodePtr>(module, "Node")
        .def_property_readonly("kind", &albert::Node::kind)
        .def_property_readonly("children", [](const albert::Node& node) {
            return node.children();
        })
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
