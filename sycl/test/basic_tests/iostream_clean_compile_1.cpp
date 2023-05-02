// RUN: %clangxx -Wno-unused-command-line-argument -Werror -fsycl -fsyntax-only %s

#include <iostream>
#include <sycl/sycl.hpp>
void foo() { std::cout << 42; }
