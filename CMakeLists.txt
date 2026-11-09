cmake_minimum_required(VERSION 3.18)
project(albert LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP QUIET)

pybind11_add_module(_core src/cpp/core.cpp src/cpp/ast.cc src/cpp/algebra.cc
                    src/cpp/canonicalize.cc src/cpp/cost.cc src/cpp/expression.cc src/cpp/hash.cc src/cpp/permutation.cc
                    src/cpp/search.cc src/cpp/simplify.cc src/cpp/parenthesize.cc
                    src/cpp/biclique.cc src/cpp/sort.cc src/cpp/cse.cc)
target_compile_features(_core PRIVATE cxx_std_17)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS _core LIBRARY DESTINATION albert)
