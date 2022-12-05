// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>
#include <type_traits>

int main() {
  static_assert(std::is_nothrow_copy_constructible_v<sycl::exception>,
                "sycl::exception is not nothrow copy constructible.");
  return 0;
}
