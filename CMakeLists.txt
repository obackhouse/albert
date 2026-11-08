cmake_minimum_required(VERSION 3.18)
project(albert LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/cpp/core.cpp src/cpp/ast.cc src/cpp/algebra.cc
                    src/cpp/search.cc)
target_compile_features(_core PRIVATE cxx_std_17)
install(TARGETS _core LIBRARY DESTINATION albert)
