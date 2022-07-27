// RUN: clang++ -Werror -fsycl %s -c

#include <iostream>
#include <sycl/sycl.hpp>
void foo() { std::cout << 42; }