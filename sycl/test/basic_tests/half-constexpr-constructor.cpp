// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
  constexpr sycl::half h;

  return 0;
}
