// RUN: %clangxx -Wno-unused-command-line-argument -Werror -fsycl %s -c

#include <iostream>
#include <sycl/sycl.hpp>
void foo() { std::cout << 42; }