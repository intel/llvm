// RUN: %clangxx -Wno-unused-command-line-argument -Werror -fsycl -fsyntax-only %s

// clang-format off
#include <sycl/sycl.hpp>
#include <iostream>
// clang-format on
void foo() { std::cout << 42; }
